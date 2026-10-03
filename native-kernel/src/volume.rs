//! Volume-based indicators (NVI, PVI, ADOSC, EFI, EMV, WAD, CVI).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::helpers::{ih_ema};

/// Calculate ADOSC (Chaikin A/D Oscillator) - Ultra-optimized version
#[pyfunction]
pub fn adosc(candles: PyReadonlyArray2<f64>, fast_period: usize, slow_period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n == 0 {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        if n == 1 {
            result[0] = 0.0;
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract price and volume data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        let volume = candles_array.column(5);
        
        // Step 1: Calculate Money Flow Multiplier and Money Flow Volume
        let mut mf_volume = Array1::<f64>::zeros(n);
        for i in 0..n {
            let price_range = high[i] - low[i];
            let multiplier = if price_range != 0.0 {
                ((close[i] - low[i]) - (high[i] - close[i])) / price_range
            } else {
                0.0
            };
            mf_volume[i] = multiplier * volume[i];
        }
        
        // Step 2: Calculate A/D Line (cumulative sum of money flow volume)
        let mut ad_line = Array1::<f64>::zeros(n);
        ad_line[0] = mf_volume[0];
        for i in 1..n {
            ad_line[i] = ad_line[i - 1] + mf_volume[i];
        }
        
        // Step 3: Calculate EMAs of A/D Line
        let fast_alpha = 2.0 / (fast_period as f64 + 1.0);
        let slow_alpha = 2.0 / (slow_period as f64 + 1.0);
        let fast_one_minus_alpha = 1.0 - fast_alpha;
        let slow_one_minus_alpha = 1.0 - slow_alpha;
        
        // Calculate fast EMA
        let mut fast_ema = Array1::<f64>::zeros(n);
        fast_ema[0] = ad_line[0];
        for i in 1..n {
            fast_ema[i] = fast_alpha * ad_line[i] + fast_one_minus_alpha * fast_ema[i - 1];
        }
        
        // Calculate slow EMA
        let mut slow_ema = Array1::<f64>::zeros(n);
        slow_ema[0] = ad_line[0];
        for i in 1..n {
            slow_ema[i] = slow_alpha * ad_line[i] + slow_one_minus_alpha * slow_ema[i - 1];
        }
        
        // Step 4: Calculate ADOSC = Fast EMA - Slow EMA
        for i in 0..n {
            result[i] = fast_ema[i] - slow_ema[i];
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// Calculate CVI (Chaikins Volatility Indicator) - Ultra-optimized version
#[pyfunction]
pub fn cvi(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut result = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok(PyArray1::from_array(py, &result).to_owned());
        }
        
        // Extract high and low price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate high-low difference
        let mut hl_diff = Array1::<f64>::zeros(n);
        for i in 0..n {
            hl_diff[i] = high[i] - low[i];
        }
        
        // Calculate EMA of the high-low difference
        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        let mut ema_diff = Array1::<f64>::zeros(n);
        ema_diff[0] = hl_diff[0];
        
        for i in 1..n {
            ema_diff[i] = alpha * hl_diff[i] + one_minus_alpha * ema_diff[i - 1];
        }
        
        // Calculate rate of change
        for i in period..n {
            if ema_diff[i - period] != 0.0 {
                result[i] = ((ema_diff[i] - ema_diff[i - period]) / ema_diff[i - period]) * 100.0;
            } else {
                result[i] = 0.0;
            }
        }
        
        Ok(PyArray1::from_array(py, &result).to_owned())
    })
}

/// EFI — Elder's Force Index
#[pyfunction]
pub fn efi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        if n < 2 { return Ok(PyArray1::from_vec(py, vec![f64::NAN; n]).to_owned()); }
        let raw: Vec<f64> = (1..n).map(|i| (src[i] - src[i-1]) * c[[i,5]]).collect();
        let ema_vals = ih_ema(&raw, period);
        let mut r = vec![f64::NAN; n];
        if n > period {
            r[period..n].copy_from_slice(&ema_vals[(period - 1)..(n - 1)]);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// EMV — Ease of Movement
#[pyfunction]
pub fn emv(candles: PyReadonlyArray2<f64>, length: usize, div: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut emv_raw = vec![0.0f64; n];
        for i in 1..n {
            let hl2_prev = (c[[i-1,3]] + c[[i-1,4]]) / 2.0;
            let hl2_curr = (c[[i,3]] + c[[i,4]]) / 2.0;
            let hl2_chg = hl2_curr - hl2_prev;
            let vol = c[[i,5]];
            emv_raw[i] = if vol != 0.0 { div * hl2_chg * (c[[i,3]] - c[[i,4]]) / vol } else { 0.0 };
        }
        let mut r = vec![0.0f64; n];
        for i in (length-1)..n {
            r[i] = emv_raw[i+1-length..=i].iter().sum::<f64>() / length as f64;
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// WAD — Williams Accumulation/Distribution
/// Note: uses candles[:,1] (open) as "close" per original code
#[pyfunction]
pub fn wad(candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut ad = vec![0.0f64; n];
        for i in 1..n {
            let close = c[[i,1]]; // WAD uses open column as "close"
            let prev_close = c[[i-1,1]];
            ad[i] = if close > prev_close {
                close - c[[i,4]].min(prev_close)
            } else if close < prev_close {
                close - c[[i,3]].max(prev_close)
            } else { 0.0 };
        }
        let mut r = vec![0.0f64; n];
        let mut cum = 0.0f64;
        for i in 0..n { cum += ad[i]; r[i] = cum; }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// NVI — Negative Volume Index
#[pyfunction]
pub fn nvi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        let mut r = vec![1000.0f64; n];
        for i in 1..n {
            if c[[i,5]] < c[[i-1,5]] {
                r[i] = r[i-1] * (1.0 + (src[i] - src[i-1]) / src[i-1]);
            } else {
                r[i] = r[i-1];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// PVI — Positive Volume Index
#[pyfunction]
pub fn pvi(source: PyReadonlyArray1<f64>, candles: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let c = candles.as_array();
        let n = src.len();
        let mut r = vec![1000.0f64; n];
        for i in 1..n {
            if c[[i,5]] > c[[i-1,5]] {
                r[i] = r[i-1] * (1.0 + (src[i] - src[i-1]) / src[i-1]);
            } else {
                r[i] = r[i-1];
            }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}
