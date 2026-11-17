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

/// numpy names of the dtypes `block_reduce` dispatches over, exported as
/// `DTYPES` so the Python layer can refuse anything else before it writes.
const DTYPES: &[&str] = &[
    "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64",
];

/// Element types the kernel dispatches over. `nan()` is `None` for integers.
trait Element: Copy + Send + Sync + PartialOrd + 'static {
    const ZERO: Self;
    /// `sum`/`mean` accumulator: an exact wider integer for integers, f64 for
    /// floats.
    type Acc: Copy + Default + std::ops::AddAssign;
    fn widen(self) -> Self::Acc;
    fn from_sum(acc: Self::Acc) -> Self;
    fn from_mean(acc: Self::Acc, count: usize) -> Self;
    fn is_nan(self) -> bool;
    fn nan() -> Option<Self>;
    /// A Python fill value as this type; an error if it cannot be one.
    fn fill_from(obj: &Bound<'_, PyAny>) -> PyResult<Self>;
}

macro_rules! impl_element_int {
    ($acc:ty: $($t:ty),*) => {$(
        impl Element for $t {
            const ZERO: Self = 0;
            type Acc = $acc;
            fn widen(self) -> $acc { self as $acc }
            // an out-of-range sum saturates rather than wraps
            fn from_sum(acc: $acc) -> Self {
                acc.clamp(<$t>::MIN as $acc, <$t>::MAX as $acc) as $t
            }
            // integer `mean` rounds half to even (numpy's rint) instead of
            // promoting to float like xarray.coarsen
            fn from_mean(acc: $acc, count: usize) -> Self {
                let n = count as $acc;
                let (q, twice_rem) = (acc.div_euclid(n), 2 * acc.rem_euclid(n));
                let up = twice_rem > n || (twice_rem == n && q & 1 == 1);
                (q + up as $acc) as $t
            }
            fn is_nan(self) -> bool { false }
            fn nan() -> Option<Self> { None }
            fn fill_from(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
                if let Ok(v) = obj.extract::<$t>() {
                    return Ok(v);
                }
                // some files declare a float _FillValue on an integer variable
                let v: f64 = obj.extract()?;
                if v.fract() == 0.0 && v >= <$t>::MIN as f64 && v <= <$t>::MAX as f64 {
                    return Ok(v as $t);
                }
                Err(PyValueError::new_err(format!(
                    "fill_value {v} is not representable as {}",
                    stringify!($t)
                )))
            }
        }
    )*};
}

macro_rules! impl_element_float {
    ($($t:ty),*) => {$(
        impl Element for $t {
            const ZERO: Self = 0.0;
            type Acc = f64;
            fn widen(self) -> f64 { self as f64 }
            fn from_sum(acc: f64) -> Self { acc as $t }
            fn from_mean(acc: f64, count: usize) -> Self { (acc / count as f64) as $t }
            fn is_nan(self) -> bool { self.is_nan() }
            fn nan() -> Option<Self> { Some(<$t>::NAN) }
            fn fill_from(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(obj.extract::<f64>()? as $t)
            }
        }
    )*};
}

// i64 holds any realistic window of <= 32-bit values and is about twice as
// fast as i128, which only the 64-bit types need
impl_element_int!(i64: u8, u16, u32, i8, i16, i32);
impl_element_int!(i128: u64, i64);
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
            let mut acc = T::Acc::default();
            let mut count = 0usize;
            for v in w {
                if skipna && is_missing(v, fill) {
                    continue;
                }
                acc += v.widen();
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
            match method {
                Method::Sum => T::from_sum(acc),
                _ => T::from_mean(acc, count),
            }
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
        // a strided view (e.g. a clamped edge tile) is copied once rather
        // than sent down the slower generic path
        let a = a.as_standard_layout();
        let src = a.as_slice().expect("standard layout is contiguous");
        return reduce_rows(src, a.shape(), &window, &out_shape, method, fill, skipna);
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
    fill_value: Option<Bound<'py, PyAny>>,
    skipna: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let method = Method::parse(method)?;
    let ndim = a.ndim();
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "array must have at least 1 dimension",
        ));
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
    if a.shape().contains(&0) {
        return Err(PyValueError::new_err(format!(
            "array must not have a zero-length axis; got shape {:?}",
            a.shape()
        )));
    }

    macro_rules! dispatch {
        ($t:ty) => {{
            let arr = a.cast::<PyArrayDyn<$t>>()?;
            let ro = arr.readonly();
            let view = ro.as_array();
            // extracted as the array's own type: via f64 a NaN fill on an
            // integer array became 0 and large i64 fills lost precision
            let fill = fill_value
                .as_ref()
                .map(<$t as Element>::fill_from)
                .transpose()?;
            let out = py.detach(|| reduce(view, &stride, method, fill, skipna));
            Ok(out.into_pyarray(py).into_any())
        }};
    }

    let dtype = a.dtype();
    if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        dispatch!(f32)
    } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        dispatch!(f64)
    } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
        dispatch!(u8)
    } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
        dispatch!(u16)
    } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
        dispatch!(u32)
    } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
        dispatch!(u64)
    } else if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
        dispatch!(i8)
    } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
        dispatch!(i16)
    } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
        dispatch!(i32)
    } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
        dispatch!(i64)
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported dtype {dtype}; expected one of {}",
            DTYPES.join(", ")
        )))
    }
}

/// CF-pack one float to a stored integer: `round((v - offset) / scale)`,
/// NaN to `fill`, out-of-range saturating. Arithmetic stays in the input
/// precision, matching numpy.
trait Pack<T>: Copy {
    fn pack_one(self, scale: Self, offset: Self, fill: T) -> T;
}

macro_rules! impl_pack {
    ($f:ty => $($t:ty),*) => {$(
        impl Pack<$t> for $f {
            #[inline]
            fn pack_one(self, scale: $f, offset: $f, fill: $t) -> $t {
                let q = (self - offset) / scale;
                // `as` casts NaN to 0, so NaN needs its own branch; it
                // saturates everything else, including the infinities
                if q.is_nan() { fill } else { q.round_ties_even() as $t }
            }
        }
    )*};
}
impl_pack!(f32 => i8, i16, i32, i64, u8, u16, u32, u64);
impl_pack!(f64 => i8, i16, i32, i64, u8, u16, u32, u64);

/// One parallel pass, no float temporary: peak memory is the input plus the
/// output.
fn pack_array<F, T>(view: ArrayViewD<F>, scale: F, offset: F, fill: T) -> ArrayD<T>
where
    F: Pack<T> + Send + Sync,
    T: Copy + Send + Sync,
{
    Zip::from(view).par_map_collect(|&v| v.pack_one(scale, offset, fill))
}

#[pyfunction]
fn pack<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyUntypedArray>,
    scale: f64,
    offset: f64,
    fill: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! to {
        ($f:ty, $t:ty) => {{
            let arr = x.cast::<PyArrayDyn<$f>>()?;
            let ro = arr.readonly();
            let view = ro.as_array();
            // extracted as the target type so large u64 fills stay exact
            let fill: $t = fill.extract()?;
            let out = py.detach(|| pack_array(view, scale as $f, offset as $f, fill));
            Ok(out.into_pyarray(py).into_any())
        }};
    }
    macro_rules! from {
        ($f:ty) => {
            match dtype {
                "int8" => to!($f, i8),
                "int16" => to!($f, i16),
                "int32" => to!($f, i32),
                "int64" => to!($f, i64),
                "uint8" => to!($f, u8),
                "uint16" => to!($f, u16),
                "uint32" => to!($f, u32),
                "uint64" => to!($f, u64),
                other => Err(PyTypeError::new_err(format!(
                    "unsupported output dtype {other}; expected an 8 to 64 bit integer"
                ))),
            }
        };
    }

    let in_dtype = x.dtype();
    if in_dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
        from!(f32)
    } else if in_dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
        from!(f64)
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported input dtype {in_dtype}; expected f32 or f64"
        )))
    }
}

#[pymodule]
fn topozarr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(block_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add("METHODS", PyTuple::new(m.py(), Method::names())?)?;
    m.add("DTYPES", PyTuple::new(m.py(), DTYPES)?)?;
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

    // values include NaN and the fill (0.0) so every skip branch runs
    fn sample(shape: &[usize]) -> ArrayD<f64> {
        let n = shape.iter().product::<usize>();
        let flat = (0..n)
            .map(|i| match (i * 7) % 6 {
                0 => f64::NAN,
                1 => 0.0,
                k => k as f64 + i as f64 * 0.37,
            })
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), flat).unwrap()
    }

    fn same(a: &ArrayD<f64>, b: &ArrayD<f64>) -> bool {
        a.shape() == b.shape()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    const CASES: [(Option<f64>, bool); 3] = [(None, true), (Some(0.0), true), (None, false)];

    // odd extents exercise trailing-partial trims
    const ND_SHAPES: [&[usize]; 5] = [
        &[7, 9],
        &[3, 7, 9],
        &[2, 3, 7, 9],
        &[2, 1, 3, 7, 9],
        &[2, 2, 1, 2, 5, 6],
    ];

    fn unit_leading(nd: usize, sy: usize, sx: usize) -> Vec<usize> {
        let mut stride = vec![1; nd];
        stride[nd - 2] = sy;
        stride[nd - 1] = sx;
        stride
    }

    #[test]
    fn row_fast_path_matches_generic() {
        for shape in ND_SHAPES {
            let a = sample(shape);
            for &(_, method) in Method::ALL {
                for (sy, sx) in [(2, 2), (3, 2), (8, 4)] {
                    let stride = unit_leading(shape.len(), sy, sx);
                    for (fill, skipna) in CASES {
                        let out = reduce(a.view(), &stride, method, fill, skipna);
                        let window: Vec<usize> =
                            shape.iter().zip(&stride).map(|(&n, &s)| s.min(n)).collect();
                        let expect =
                            reduce_generic(a.view(), &window, out.shape(), method, fill, skipna);
                        assert!(
                            same(&out, &expect),
                            "{shape:?} {method:?} {stride:?} {fill:?} {skipna}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nd_matches_folded_3d() {
        // leading unit-stride axes are independent, so folding them is a no-op
        for shape in [&[2usize, 3, 2, 8, 10][..], &[2, 1, 3, 2, 7, 9]] {
            let nd = shape.len();
            let a = sample(shape);
            let lead = shape[..nd - 2].iter().product::<usize>();
            let folded = a
                .view()
                .into_shape_with_order(IxDyn(&[lead, shape[nd - 2], shape[nd - 1]]))
                .unwrap();
            for &(_, method) in Method::ALL {
                for (fill, skipna) in CASES {
                    let out = reduce(a.view(), &unit_leading(nd, 2, 2), method, fill, skipna);
                    let f = reduce(folded.view(), &[1, 2, 2], method, fill, skipna);
                    let f = f.into_shape_with_order(IxDyn(out.shape())).unwrap();
                    assert!(same(&out, &f), "{shape:?} {method:?} {fill:?} {skipna}");
                }
            }
        }
    }

    #[test]
    fn nd_generic_with_leading_stride() {
        // stride > 1 on leading axes forces the generic path; compare against a
        // per-output-cell window slice
        let shape = [3usize, 4, 2, 7, 9];
        let a = sample(&shape);
        for stride in [[2usize, 1, 1, 2, 2], [3, 2, 2, 3, 2], [1, 4, 1, 8, 4]] {
            let window: Vec<usize> = shape.iter().zip(&stride).map(|(&n, &s)| s.min(n)).collect();
            for &(_, method) in Method::ALL {
                for (fill, skipna) in CASES {
                    let out = reduce(a.view(), &stride, method, fill, skipna);
                    let expect = ArrayD::from_shape_fn(IxDyn(out.shape()), |o| {
                        let w = a.slice_each_axis(|ax| {
                            let k = ax.axis.index();
                            let s = o[k] * window[k];
                            ndarray::Slice::from(s..s + window[k])
                        });
                        reduce_window(w.iter().copied(), method, fill, skipna)
                    });
                    assert!(
                        same(&out, &expect),
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
    fn integer_sum_and_mean_are_exact_past_f64() {
        // 2^53 + 1 is not an f64; an f64 accumulator returned 2^53
        let big = (1i64 << 53) + 1;
        let a = array![[big, big], [big, big]].into_dyn();
        assert_eq!(
            reduce(a.view(), &[2, 2], Method::Mean, None, true)[[0, 0]],
            big
        );
        let b = array![[big, 0], [0, 0]].into_dyn();
        assert_eq!(
            reduce(b.view(), &[2, 2], Method::Sum, None, true)[[0, 0]],
            big
        );
        // sums saturate at the dtype bounds, as before
        let c = array![[200u8, 200], [200, 200]].into_dyn();
        assert_eq!(
            reduce(c.view(), &[2, 2], Method::Sum, None, true)[[0, 0]],
            255
        );
        let d = array![[i64::MIN, i64::MIN], [-1, 0]].into_dyn();
        assert_eq!(
            reduce(d.view(), &[2, 2], Method::Sum, None, true)[[0, 0]],
            i64::MIN
        );
    }

    #[test]
    fn integer_mean_rounds_half_to_even_both_signs() {
        for (vals, want) in [
            ([1i32, 2, 1, 2], 2),   // 1.5 -> 2
            ([2, 3, 2, 3], 2),      // 2.5 -> 2
            ([-1, -2, -1, -2], -2), // -1.5 -> -2
            ([-2, -3, -2, -3], -2), // -2.5 -> -2
            ([0, 0, 0, 1], 0),      // 0.25 -> 0
            ([-1, 0, 0, 0], 0),     // -0.25 -> 0
            ([-1, -1, -1, 0], -1),  // -0.75 -> -1
        ] {
            let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vals.to_vec()).unwrap();
            assert_eq!(
                reduce(a.view(), &[2, 2], Method::Mean, None, true)[[0, 0]],
                want,
                "{vals:?}"
            );
        }
    }

    #[test]
    fn strided_view_takes_the_row_path() {
        // a non-contiguous view is copied to standard layout, not rejected
        let a = sample(&[3, 8, 10]);
        let v = a.slice_each_axis(|_| ndarray::Slice::new(0, None, 2));
        let out = reduce(v.view(), &[1, 2, 2], Method::Mean, None, true);
        let expect = reduce(v.to_owned().view(), &[1, 2, 2], Method::Mean, None, true);
        assert!(same(&out, &expect));
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

    #[test]
    fn pack_nan_inf_and_range_saturate() {
        let x = array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e9, -1e9, 12.0].into_dyn();
        let out: ArrayD<i16> = pack_array(x.view(), 1.0, 0.0, -9999);
        assert_eq!(
            out.as_slice().unwrap(),
            &[-9999, i16::MAX, i16::MIN, i16::MAX, i16::MIN, 12]
        );
        let top: ArrayD<i64> = pack_array(array![1e30, -1e30].into_dyn().view(), 1.0, 0.0, 0);
        assert_eq!(top.as_slice().unwrap(), &[i64::MAX, i64::MIN]);
    }

    #[test]
    fn pack_rounds_ties_to_even_after_scale_offset() {
        let x = array![10.5, 11.5, 12.5, 9.5].into_dyn();
        let out: ArrayD<u8> = pack_array(x.view(), 1.0, 10.0, 255);
        assert_eq!(out.as_slice().unwrap(), &[0, 2, 2, 0]);
        let scaled: ArrayD<i32> = pack_array(array![0.25, 0.75].into_dyn().view(), 0.5, 0.0, 0);
        assert_eq!(scaled.as_slice().unwrap(), &[0, 2]);
    }

    #[test]
    fn pack_strided_view_keeps_logical_order() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out: ArrayD<i16> = pack_array(x.t().into_dyn(), 1.0, 0.0, 0);
        assert_eq!(out, array![[1i16, 4], [2, 5], [3, 6]].into_dyn());
    }

    #[test]
    fn pack_f32_stays_in_f32() {
        // 0.35f32 / 0.1f32 is the tie 3.5 in f32 (packs to 4); widened to
        // f64 the same inputs give 3.4999999 (packs to 3)
        let (v, scale) = (0.35f32, 0.1f32);
        assert_eq!(((v as f64) / (scale as f64)).round_ties_even() as i16, 3);
        let out: ArrayD<i16> = pack_array(array![v].into_dyn().view(), scale, 0.0, 0);
        assert_eq!(out[[0]], 4);
    }
}
