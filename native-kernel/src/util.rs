//! Miscellaneous non-indicator utilities exposed to Python.

use ndarray::{s, Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rust_decimal::Decimal;
use std::str::FromStr;

/// Shift array by a given number of periods
#[pyfunction]
pub fn shift(source: PyReadonlyArray1<f64>, periods: isize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if periods == 0 {
            // No shift, return copy of source
            for i in 0..n {
                result[i] = source_array[i];
            }
        } else if periods > 0 {
            // Shift right (positive periods)
            let shift_amount = periods as usize;
            if shift_amount < n {
                for i in shift_amount..n {
                    result[i] = source_array[i - shift_amount];
                }
            }
        } else {
            // Shift left (negative periods)
            let shift_amount = (-periods) as usize;
            if shift_amount < n {
                for i in 0..(n - shift_amount) {
                    result[i] = source_array[i + shift_amount];
                }
            }
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate moving standard deviation
#[pyfunction]
pub fn moving_std(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);

        if n < period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }

        // Calculate moving standard deviation
        for i in (period - 1)..n {
            let start_idx = i + 1 - period;
            let window = source_array.slice(s![start_idx..=i]);
            
            // Calculate mean
            let mean = window.sum() / period as f64;
            
            // Calculate variance
            let variance = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / period as f64;
            
            // Calculate standard deviation
            result[i] = variance.sqrt();
        }

        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Sums two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn sum_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise addition
    let result_decimal = decimal1 + decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}

/// Subtracts two floats without the rounding issue by using precise decimal arithmetic
#[pyfunction]
pub fn subtract_floats(float1: f64, float2: f64) -> PyResult<f64> {
    // Convert floats to Decimal for precise arithmetic
    let decimal1 = Decimal::from_str(&float1.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float1: {}", e)))?;
    let decimal2 = Decimal::from_str(&float2.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid float2: {}", e)))?;
    
    // Perform precise subtraction
    let result_decimal = decimal1 - decimal2;
    
    // Convert back to f64
    let result = result_decimal.to_string().parse::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Result conversion error: {}", e)))?;
    
    Ok(result)
}

/// Find index of a row in a 2D array that matches the given 1D array exactly.
/// Returns -1 if no match. Mirrors np.all(orders[i] == order_array) used by FuturesExchange.
#[pyfunction]
pub fn find_order_index(
    orders: PyReadonlyArray2<f64>,
    order_array: PyReadonlyArray1<f64>,
) -> PyResult<i64> {
    let orders = orders.as_array();
    let target = order_array.as_array();
    let n_rows = orders.shape()[0];
    let n_cols = orders.shape()[1];
    if target.len() != n_cols { return Ok(-1); }
    'outer: for i in 0..n_rows {
        for j in 0..n_cols {
            if orders[[i, j]] != target[j] { continue 'outer; }
        }
        return Ok(i as i64);
    }
    Ok(-1)
}
