//! Statistical indicators (linear regression, Hurst exponent, Damiani volatmeter).

use ndarray::s;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::types::{PyArrTuple2};

use crate::helpers::{ih_atr_wilder, rolling_std_pop};

/// Linear Regression Value
#[pyfunction]
pub fn linearreg(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n < period { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        let mean_x = (period as f64 - 1.0) / 2.0;
        let sxx: f64 = (0..period).map(|j| (j as f64 - mean_x).powi(2)).sum();
        for i in (period-1)..n {
            let win = &src.as_slice().unwrap()[i+1-period..=i];
            let mean_y = win.iter().sum::<f64>() / period as f64;
            let sxy: f64 = win.iter().enumerate().map(|(j, &y)| (y - mean_y) * (j as f64 - mean_x)).sum();
            r[i] = mean_y + mean_x * (sxy / sxx);
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Hurst RS (R/S method for Hurst exponent)
#[pyfunction]
pub fn hurst_rs(x: PyReadonlyArray1<f64>, min_chunksize: usize, max_chunksize: usize, num_chunksize: usize) -> PyResult<f64> {
    Python::with_gil(|_py| {
        let x_arr = x.as_array();
        let xv: Vec<f64> = x_arr.to_vec();
        let n = xv.len();
        let max_cs = max_chunksize + 1;
        let step = if num_chunksize > 1 { (max_cs - min_chunksize) as f64 / (num_chunksize - 1) as f64 } else { 1.0 };
        let chunk_sizes: Vec<usize> = (0..num_chunksize).map(|i| (min_chunksize as f64 + i as f64 * step) as usize).collect();
        let mut rs_values = vec![0.0f64; num_chunksize];
        for (ci, &cs) in chunk_sizes.iter().enumerate() {
            let nchunks = n / cs;
            let mut rs_sum = 0.0f64;
            let mut valid = 0usize;
            for idx in 0..nchunks {
                let chunk = &xv[idx*cs..(idx+1)*cs];
                let mean = chunk.iter().sum::<f64>() / cs as f64;
                let mut cum = 0.0f64;
                let mut z = Vec::with_capacity(cs);
                for &v in chunk { cum += v - mean; z.push(cum); }
                let r_val = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                          - z.iter().cloned().fold(f64::INFINITY, f64::min);
                let var: f64 = chunk.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cs as f64;
                let s = var.sqrt();
                if s > 0.0 { rs_sum += r_val / s; valid += 1; }
            }
            rs_values[ci] = if valid > 0 { rs_sum / valid as f64 } else { 0.0 };
        }
        // Linear regression: log(rs) = H*log(chunk_size) + c
        let log_cs: Vec<f64> = chunk_sizes.iter().map(|&x| (x as f64).ln()).collect();
        let log_rs: Vec<f64> = rs_values.iter().map(|&x| if x > 0.0 { x.ln() } else { 0.0 }).collect();
        let m = num_chunksize as f64;
        let sx: f64 = log_cs.iter().sum();
        let sy: f64 = log_rs.iter().sum();
        let sxx: f64 = log_cs.iter().map(|&x| x*x).sum();
        let sxy: f64 = log_cs.iter().zip(log_rs.iter()).map(|(&x,&y)| x*y).sum();
        let h = (m*sxy - sx*sy) / (m*sxx - sx*sx);
        Ok(h)
    })
}

/// Damiani Volatmeter — returns (vol, anti)
#[pyfunction]
pub fn damiani_volatmeter(
    candles: PyReadonlyArray2<f64>,
    source: PyReadonlyArray1<f64>,
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let src: Vec<f64> = source.as_array().to_vec();
        let n = candles_array.shape()[0];
        let lag_s = 0.5f64;

        let high: Vec<f64> = candles_array.slice(s![.., 3]).to_vec();
        let low: Vec<f64> = candles_array.slice(s![.., 4]).to_vec();
        let close: Vec<f64> = candles_array.slice(s![.., 2]).to_vec();

        let atrvis = ih_atr_wilder(&high, &low, &close, vis_atr);
        let atrsed = ih_atr_wilder(&high, &low, &close, sed_atr);

        // u[i] = atrvis[i] / atrsed[i] for i >= sed_std
        let mut u = vec![0.0f64; n];
        for i in sed_std..n {
            if atrsed[i] != 0.0 && !atrsed[i].is_nan() && !atrvis[i].is_nan() {
                u[i] = atrvis[i] / atrsed[i];
            }
        }

        // lfilter with b=[1.0], a=[1.0, -0.5, 0.0, 0.5]:
        //   y[i] = u[i] + 0.5*y[i-1] - 0.5*y[i-3]
        let mut vol = vec![0.0f64; n];
        for i in 0..n {
            let mut y = u[i];
            if i >= 1 { y += lag_s * vol[i - 1]; }
            if i >= 3 { y -= lag_s * vol[i - 3]; }
            vol[i] = y;
        }

        // Rolling std and threshold calculation
        let mut t = vec![0.0f64; n];
        if n >= sed_std {
            let std_vis = rolling_std_pop(&src, vis_std);
            let std_sed = rolling_std_pop(&src, sed_std);
            for idx in sed_std..n {
                let v = std_vis[idx - 1];
                let s = std_sed[idx - 1];
                t[idx] = threshold - v / s;
            }
        }

        Ok((
            PyArray1::from_vec(py, vol).to_owned(),
            PyArray1::from_vec(py, t).to_owned(),
        ))
    })
}
