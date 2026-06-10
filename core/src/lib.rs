use ndarray::{ArrayD, ArrayViewD, IxDyn, Zip};
use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Mean,
    Max,
    Min,
    Sum,
}

impl Method {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "mean" => Ok(Method::Mean),
            "max" => Ok(Method::Max),
            "min" => Ok(Method::Min),
            "sum" => Ok(Method::Sum),
            _ => Err(PyValueError::new_err(format!(
                "method must be one of 'mean', 'max', 'min', 'sum'; got {s:?}"
            ))),
        }
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

fn reduce_window<T: Element>(
    w: &ArrayViewD<T>,
    method: Method,
    fill: Option<T>,
    skipna: bool,
) -> T {
    match method {
        Method::Mean | Method::Sum => {
            let mut acc = 0.0f64;
            let mut count = 0usize;
            for &v in w.iter() {
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
            for &v in w.iter() {
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

    let mut out = ArrayD::<T>::from_elem(IxDyn(&out_shape), T::ZERO);
    Zip::from(&mut out)
        .and(a.exact_chunks(IxDyn(&window)))
        .par_for_each(|o, w| *o = reduce_window(&w.into_dyn(), method, fill, skipna));
    out
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
    Ok(())
}
