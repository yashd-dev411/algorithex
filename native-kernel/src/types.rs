//! Shared type aliases for the PyO3 return types used across the crate.
//!
//! Most indicators return one or more 1-D numpy arrays. The raw signatures
//! (`PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, …)>`) are noisy and
//! triggered ~20 `clippy::type_complexity` warnings; these aliases keep the
//! signatures readable.

use numpy::PyArray1;
use pyo3::prelude::*;

/// A heap-rooted, owned 1-D `f64` numpy array.
pub type PyArr = Py<PyArray1<f64>>;

/// Result of an indicator returning two parallel 1-D series.
pub type PyArrTuple2 = PyResult<(PyArr, PyArr)>;

/// Result of an indicator returning three parallel 1-D series.
pub type PyArrTuple3 = PyResult<(PyArr, PyArr, PyArr)>;

/// Result of an indicator returning four parallel 1-D series.
pub type PyArrTuple4 = PyResult<(PyArr, PyArr, PyArr, PyArr)>;
