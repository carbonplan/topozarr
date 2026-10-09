use ndarray::{ArrayD, ArrayViewD, IxDyn, Zip};
use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Mean,
    Max,
    Min,
    Sum,
    Nearest,
}

impl Method {
    /// The one place a coarsening method is named. Drives `parse`, the error
    /// message, and the `METHODS` tuple the Python layer validates against, so
    /// a new variant cannot be half-added.
    const ALL: &'static [(&'static str, Method)] = &[
        ("mean", Method::Mean),
        ("max", Method::Max),
        ("min", Method::Min),
        ("sum", Method::Sum),
        ("nearest", Method::Nearest),
    ];

    fn names() -> Vec<&'static str> {
        Self::ALL.iter().map(|(name, _)| *name).collect()
    }

    /// `'mean', 'max', ...` -- the method list as it appears in errors.
    fn names_display() -> String {
        Self::names()
            .iter()
            .map(|n| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn parse(s: &str) -> PyResult<Self> {
        Self::ALL
            .iter()
            .find(|(name, _)| *name == s)
            .map(|(_, method)| *method)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "method must be one of {}; got {s:?}",
                    Self::names_display()
                ))
            })
    }
}

/// Element types the kernel dispatches over. `nan()` is `None` for integers.
trait Element: Copy + Send + Sync + PartialOrd + 'static {
    const ZERO: Self;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn is_nan(self) -> bool;
    fn nan() -> Option<Self>;
}

macro_rules! impl_element_int {
    ($($t:ty),*) => {$(
        impl Element for $t {
            const ZERO: Self = 0;
            fn to_f64(self) -> f64 { self as f64 }
            // `as` truncates toward zero, so integer `mean` floors the window
            // average instead of promoting to float like xarray.coarsen
            fn from_f64(v: f64) -> Self { v as $t }
            fn is_nan(self) -> bool { false }
            fn nan() -> Option<Self> { None }
        }
    )*};
}

macro_rules! impl_element_float {
    ($($t:ty),*) => {$(
        impl Element for $t {
            const ZERO: Self = 0.0;
            fn to_f64(self) -> f64 { self as f64 }
            fn from_f64(v: f64) -> Self { v as $t }
            fn is_nan(self) -> bool { self.is_nan() }
            fn nan() -> Option<Self> { Some(<$t>::NAN) }
        }
    )*};
}

impl_element_int!(u8, u16, i16, i32, i64);
impl_element_float!(f32, f64);

#[inline]
fn is_missing<T: Element>(v: T, fill: Option<T>) -> bool {
    v.is_nan() || fill.is_some_and(|f| v == f)
}

/// Value emitted for a window with zero valid elements.
/// Unreachable for integer dtypes without a fill value (nothing can be missing).
#[inline]
fn all_missing_result<T: Element>(fill: Option<T>) -> T {
    fill.or_else(T::nan).unwrap_or(T::ZERO)
}

/// Reduce one window, given its elements in row-major order.
fn reduce_window<T: Element>(
    mut w: impl Iterator<Item = T>,
    method: Method,
    fill: Option<T>,
    skipna: bool,
) -> T {
    match method {
        // corner-pick decimation: fill/skipna do not apply, which keeps the
        // op exactly composable (corner-of-corners == corner-of-native)
        Method::Nearest => w.next().expect("windows are never empty"),
        Method::Mean | Method::Sum => {
            let mut acc = 0.0f64;
            let mut count = 0usize;
            for v in w {
                if skipna && is_missing(v, fill) {
                    continue;
                }
                acc += v.to_f64();
                count += 1;
            }
            if count == 0 {
                // sum over an all-missing window is 0, matching numpy nansum
                // and xarray's skipna sum; mean is fill/NaN
                return match method {
                    Method::Sum => T::ZERO,
                    _ => all_missing_result(fill),
                };
            }
            if method == Method::Mean {
                acc /= count as f64;
            }
            T::from_f64(acc)
        }
        Method::Max | Method::Min => {
            let mut best: Option<T> = None;
            for v in w {
                if skipna {
                    if is_missing(v, fill) {
                        continue;
                    }
                } else if v.is_nan() {
                    return v;
                }
                best = Some(match best {
                    None => v,
                    Some(b) => {
                        let take = if method == Method::Max { v > b } else { v < b };
                        if take {
                            v
                        } else {
                            b
                        }
                    }
                });
            }
            best.unwrap_or_else(|| all_missing_result(fill))
        }
    }
}

fn reduce<T: Element>(
    a: ArrayViewD<T>,
    stride: &[usize],
    method: Method,
    fill: Option<T>,
    skipna: bool,
) -> ArrayD<T> {
    // Output: max(n // s, 1) per axis. Windows are min(s, n) wide so an axis
    // smaller than its stride still yields one window; exact_chunks drops
    // trailing partial windows, reproducing coarsen(boundary="trim").
    let out_shape: Vec<usize> = a
        .shape()
        .iter()
        .zip(stride)
        .map(|(&n, &s)| (n / s).max(1))
        .collect();
    let window: Vec<usize> = a
        .shape()
        .iter()
        .zip(stride)
        .map(|(&n, &s)| s.min(n).max(1))
        .collect();

    let nd = a.ndim();
    if nd >= 2 && stride[..nd - 2].iter().all(|&s| s == 1) {
        if let Some(src) = a.as_slice() {
            return reduce_rows(src, a.shape(), &window, &out_shape, method, fill, skipna);
        }
    }
    reduce_generic(a, &window, &out_shape, method, fill, skipna)
}

fn reduce_generic<T: Element>(
    a: ArrayViewD<T>,
    window: &[usize],
    out_shape: &[usize],
    method: Method,
    fill: Option<T>,
    skipna: bool,
) -> ArrayD<T> {
    let mut out = ArrayD::<T>::from_elem(IxDyn(out_shape), T::ZERO);
    Zip::from(&mut out)
        .and(a.exact_chunks(IxDyn(window)))
        .par_for_each(|o, w| *o = reduce_window(w.iter().copied(), method, fill, skipna));
    out
}

/// Fast path for C-contiguous input reduced over the last two axes only:
/// windows are read as row slices, and rayon splits by output row.
fn reduce_rows<T: Element>(
    src: &[T],
    shape: &[usize],
    window: &[usize],
    out_shape: &[usize],
    method: Method,
    fill: Option<T>,
    skipna: bool,
) -> ArrayD<T> {
    let nd = shape.len();
    let (h, w) = (shape[nd - 2], shape[nd - 1]);
    let (sy, sx) = (window[nd - 2], window[nd - 1]);
    let (oh, ow) = (out_shape[nd - 2], out_shape[nd - 1]);
    let mut out = vec![T::ZERO; out_shape.iter().product()];
    out.par_chunks_mut(ow).enumerate().for_each(|(r, row)| {
        let top = (r / oh) * h + (r % oh) * sy;
        for (ox, o) in row.iter_mut().enumerate() {
            let x0 = ox * sx;
            let win =
                (top..top + sy).flat_map(|y| src[y * w + x0..y * w + x0 + sx].iter().copied());
            *o = reduce_window(win, method, fill, skipna);
        }
    });
    ArrayD::from_shape_vec(IxDyn(out_shape), out).expect("out_shape matches buffer")
}

#[pyfunction]
#[pyo3(signature = (a, stride, method, fill_value=None, skipna=true))]
fn block_reduce<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyUntypedArray>,
    stride: Vec<usize>,
    method: &str,
    fill_value: Option<f64>,
    skipna: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let method = Method::parse(method)?;
    let ndim = a.ndim();
    if ndim == 0 || ndim > 4 {
        return Err(PyValueError::new_err(format!(
            "array must have 1-4 dimensions, got {ndim}"
        )));
    }
    if stride.len() != ndim {
        return Err(PyValueError::new_err(format!(
            "stride length {} does not match array ndim {ndim}",
            stride.len()
        )));
    }
    if stride.contains(&0) {
        return Err(PyValueError::new_err("stride entries must be >= 1"));
    }

    macro_rules! dispatch {
        ($t:ty) => {{
            let arr = a.cast::<PyArrayDyn<$t>>()?;
            let ro = arr.readonly();
            let view = ro.as_array();
            let fill = fill_value.map(<$t as Element>::from_f64);
            let out = py.detach(|| reduce(view, &stride, method, fill, skipna));
            Ok(out.into_pyarray(py).into_any())
        }};
    }

    let dtype = a.dtype();
    if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        dispatch!(f32)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        dispatch!(f64)
    } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        dispatch!(i16)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        dispatch!(i32)
    } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
        dispatch!(i64)
    } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
        dispatch!(u8)
    } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
        dispatch!(u16)
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported dtype {dtype}; expected one of u8, u16, i16, i32, i64, f32, f64"
        )))
    }
}

#[pymodule]
fn topozarr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(block_reduce, m)?)?;
    m.add("METHODS", PyTuple::new(m.py(), Method::names())?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn methods_names_match_parse() {
        // every exported name parses, and nothing else does
        for name in Method::names() {
            assert!(Method::parse(name).is_ok(), "{name} does not parse");
        }
        assert_eq!(Method::names().len(), Method::ALL.len());
        assert!(Method::parse("median").is_err());
    }

    #[test]
    fn error_text_lists_every_method() {
        // PyErr::to_string needs an interpreter, so assert on the message
        // fragment parse() embeds rather than on the error itself
        let listed = Method::names_display();
        for name in Method::names() {
            assert!(listed.contains(name), "error text omits {name}: {listed}");
        }
    }

    #[test]
    fn row_fast_path_matches_generic() {
        // odd extents exercise trailing-partial trims; values include NaN and
        // the fill so every skip branch runs
        let shape = [3usize, 7, 9];
        let a = ArrayD::from_shape_fn(IxDyn(&shape), |i| match (i[0] * 31 + i[1] * 7 + i[2]) % 6 {
            0 => f64::NAN,
            1 => 0.0,
            k => k as f64 + i[2] as f64 * 0.37,
        });
        for &(_, method) in Method::ALL {
            for stride in [[1usize, 2, 2], [1, 3, 2], [1, 8, 4]] {
                for (fill, skipna) in [(None, true), (Some(0.0), true), (None, false)] {
                    let out = reduce(a.view(), &stride, method, fill, skipna);
                    let window: Vec<usize> =
                        shape.iter().zip(&stride).map(|(&n, &s)| s.min(n)).collect();
                    let expect =
                        reduce_generic(a.view(), &window, out.shape(), method, fill, skipna);
                    assert!(
                        out.iter()
                            .zip(expect.iter())
                            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
                        "{method:?} {stride:?} {fill:?} {skipna}"
                    );
                }
            }
        }
    }

    #[test]
    fn methods_basic_2x2() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]].into_dyn();
        for (method, expected) in [
            (Method::Mean, 2.5),
            (Method::Max, 4.0),
            (Method::Min, 1.0),
            (Method::Sum, 10.0),
        ] {
            let out = reduce(a.view(), &[2, 2], method, None, true);
            assert_eq!(out.shape(), &[1, 1]);
            assert_eq!(out[[0, 0]], expected);
        }
    }

    #[test]
    fn trims_trailing_partial_windows() {
        // 3x3 with stride 2: only the complete top-left 2x2 window survives,
        // matching coarsen(boundary="trim")
        let a = array![[1.0f64, 2.0, 9.0], [3.0, 4.0, 9.0], [9.0, 9.0, 9.0]].into_dyn();
        let out = reduce(a.view(), &[2, 2], Method::Mean, None, true);
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out[[0, 0]], 2.5);
    }

    #[test]
    fn axis_smaller_than_stride_yields_one_window() {
        let a = array![[1.0f64, 2.0, 3.0, 4.0]].into_dyn();
        let out = reduce(a.view(), &[2, 2], Method::Sum, None, true);
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out[[0, 0]], 3.0);
        assert_eq!(out[[0, 1]], 7.0);
    }

    #[test]
    fn nan_handling_per_skipna() {
        let a = array![[1.0f64, f64::NAN], [3.0, 5.0]].into_dyn();
        let out = reduce(a.view(), &[2, 2], Method::Mean, None, true);
        assert_eq!(out[[0, 0]], 3.0);
        let out = reduce(a.view(), &[2, 2], Method::Mean, None, false);
        assert!(out[[0, 0]].is_nan());
        let out = reduce(a.view(), &[2, 2], Method::Max, None, false);
        assert!(out[[0, 0]].is_nan());
    }

    #[test]
    fn integer_fill_value_skipped() {
        let a = array![[1u8, 255], [3, 5]].into_dyn();
        let out = reduce(a.view(), &[2, 2], Method::Mean, Some(255), true);
        assert_eq!(out[[0, 0]], 3); // (1 + 3 + 5) / 3
        let out = reduce(a.view(), &[2, 2], Method::Min, Some(255), true);
        assert_eq!(out[[0, 0]], 1);
    }

    #[test]
    fn all_missing_window_semantics() {
        // sum -> 0 (nansum); mean/max -> fill when given, else NaN
        let a = array![[f64::NAN, f64::NAN], [f64::NAN, f64::NAN]].into_dyn();
        assert_eq!(
            reduce(a.view(), &[2, 2], Method::Sum, None, true)[[0, 0]],
            0.0
        );
        assert!(reduce(a.view(), &[2, 2], Method::Mean, None, true)[[0, 0]].is_nan());
        assert!(reduce(a.view(), &[2, 2], Method::Max, None, true)[[0, 0]].is_nan());
        let out = reduce(a.view(), &[2, 2], Method::Mean, Some(-9999.0), true);
        assert_eq!(out[[0, 0]], -9999.0);

        let b = array![[7i32, 7], [7, 7]].into_dyn();
        assert_eq!(
            reduce(b.view(), &[2, 2], Method::Min, Some(7), true)[[0, 0]],
            7
        );
        assert_eq!(
            reduce(b.view(), &[2, 2], Method::Sum, Some(7), true)[[0, 0]],
            0
        );
    }

    #[test]
    fn nearest_picks_window_corner() {
        let a = array![[1.0f64, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 9.0]].into_dyn();
        // stride 2 on 3x3 trims to one window; corner is [0, 0]
        let out = reduce(a.view(), &[2, 2], Method::Nearest, None, true);
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out[[0, 0]], 1.0);

        let b = array![[1i32, 2, 3, 4], [5, 6, 7, 8]].into_dyn();
        let out = reduce(b.view(), &[2, 2], Method::Nearest, None, true);
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out[[0, 0]], 1);
        assert_eq!(out[[0, 1]], 3);
    }

    #[test]
    fn nearest_ignores_fill_and_nan() {
        // fill/NaN corners pass through unchanged regardless of skipna
        let a = array![[f64::NAN, 2.0], [3.0, 4.0]].into_dyn();
        assert!(reduce(a.view(), &[2, 2], Method::Nearest, None, true)[[0, 0]].is_nan());
        let b = array![[255u8, 2], [3, 4]].into_dyn();
        assert_eq!(
            reduce(b.view(), &[2, 2], Method::Nearest, Some(255), true)[[0, 0]],
            255
        );
    }

    #[test]
    fn nearest_composes_across_steps() {
        // 2x then 2x equals 4x directly (corner-of-corners == corner-of-native)
        let a = ArrayD::from_shape_fn(IxDyn(&[8, 8]), |ix| (ix[0] * 8 + ix[1]) as f64);
        let two_step = reduce(
            reduce(a.view(), &[2, 2], Method::Nearest, None, true).view(),
            &[2, 2],
            Method::Nearest,
            None,
            true,
        );
        let one_step = reduce(a.view(), &[4, 4], Method::Nearest, None, true);
        assert_eq!(two_step, one_step);
    }

    #[test]
    fn three_d_unit_stride_on_leading_axis() {
        let a = array![[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]].into_dyn();
        let out = reduce(a.view(), &[1, 2, 2], Method::Mean, None, true);
        assert_eq!(out.shape(), &[2, 1, 1]);
        assert_eq!(out[[0, 0, 0]], 2.5);
        assert_eq!(out[[1, 0, 0]], 25.0);
    }
}
