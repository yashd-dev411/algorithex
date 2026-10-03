//! Trend / directional indicators (ADX family, SAR, Supertrend, alligator).

use ndarray::{s, Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::types::{PyArrTuple2, PyArrTuple3};

use crate::helpers::{ih_atr_wilder};

/// Calculate Ichimoku Cloud
#[pyfunction]
pub fn ichimoku_cloud(
    candles: PyReadonlyArray2<f64>,
    conversion_line_period: usize,
    base_line_period: usize,
    lagging_line_period: usize,
    displacement: usize
) -> PyResult<(f64, f64, f64, f64)> {
    Python::with_gil(|_py| {
        let candles_array = candles.as_array();
        
        // Get the high and low price columns
        let high_prices = candles_array.slice(s![.., 3]);
        let low_prices = candles_array.slice(s![.., 4]);
        
        // Calculate for earlier period (displaced)
        let earlier_high = high_prices.slice(s![..-((displacement as isize) - 1)]);
        let earlier_low = low_prices.slice(s![..-((displacement as isize) - 1)]);
        
        // Helper function to get period high and low
        let get_period_hl = |highs: ndarray::ArrayView1<f64>, lows: ndarray::ArrayView1<f64>, period: usize| -> (f64, f64) {
            let n = highs.len();
            if n < period {
                return (f64::NAN, f64::NAN);
            }
            
            // Instead of negative slicing, use the last 'period' elements
            let start_idx = n.saturating_sub(period);
            let period_high = highs.slice(s![start_idx..]).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low = lows.slice(s![start_idx..]).fold(f64::INFINITY, |a, &b| a.min(b));
            
            (period_high, period_low)
        };
        
        // Earlier periods calculations
        let (small_ph, small_pl) = get_period_hl(earlier_high, earlier_low, conversion_line_period);
        let (mid_ph, mid_pl) = get_period_hl(earlier_high, earlier_low, base_line_period);
        let (long_ph, long_pl) = get_period_hl(earlier_high, earlier_low, lagging_line_period);
        
        let early_conversion_line = (small_ph + small_pl) / 2.0;
        let early_base_line = (mid_ph + mid_pl) / 2.0;
        let span_a = (early_conversion_line + early_base_line) / 2.0;
        let span_b = (long_ph + long_pl) / 2.0;
        
        // Current period calculations
        let (current_small_ph, current_small_pl) = get_period_hl(high_prices, low_prices, conversion_line_period);
        let (current_mid_ph, current_mid_pl) = get_period_hl(high_prices, low_prices, base_line_period);
        
        let current_conversion_line = (current_small_ph + current_small_pl) / 2.0;
        let current_base_line = (current_mid_ph + current_mid_pl) / 2.0;
        
        Ok((current_conversion_line, current_base_line, span_a, span_b))
    })
}

/// Calculate ADX (Average Directional Index)
#[pyfunction]
pub fn adx(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut adx_result = Array1::<f64>::from_elem(n, f64::NAN);

        let required_len = 2 * period;
        if n <= required_len {
            return Ok(PyArray1::from_array(py, &adx_result).to_owned());
        }

        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);

        // State for Wilder smoothing
        let mut tr_smooth: f64 = 0.0;
        let mut plus_dm_smooth: f64 = 0.0;
        let mut minus_dm_smooth: f64 = 0.0;
        
        // Buffer for DX values to calculate the first ADX
        let mut dx_buffer: Vec<f64> = Vec::with_capacity(period);

        // Main calculation loop
        for i in 1..n {
            // 1. Calculate raw TR, +DM, -DM for current step `i`
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);

            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];

            let mut current_plus_dm = 0.0;
            if h_diff > l_diff && h_diff > 0.0 {
                current_plus_dm = h_diff;
            }

            let mut current_minus_dm = 0.0;
            if l_diff > h_diff && l_diff > 0.0 {
                current_minus_dm = l_diff;
            }

            // 2. Update smoothed values
            if i <= period {
                // Accumulate for the first smoothed value
                tr_smooth += current_tr;
                plus_dm_smooth += current_plus_dm;
                minus_dm_smooth += current_minus_dm;
            } else {
                // Apply Wilder's smoothing formula
                tr_smooth = tr_smooth - (tr_smooth / period as f64) + current_tr;
                plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period as f64) + current_plus_dm;
                minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period as f64) + current_minus_dm;
            }
            
            // From index `period` onwards, we can calculate DI and DX
            if i >= period {
                let mut current_dx = 0.0;
                if tr_smooth != 0.0 {
                    let di_plus = 100.0 * plus_dm_smooth / tr_smooth;
                    let di_minus = 100.0 * minus_dm_smooth / tr_smooth;
                    let di_sum = di_plus + di_minus;
                    if di_sum != 0.0 {
                        current_dx = 100.0 * (di_plus - di_minus).abs() / di_sum;
                    }
                }
                
                // Store DX value for initial ADX calculation, or calculate ADX
                if i < required_len {
                    dx_buffer.push(current_dx);
                } else if i == required_len {
                    // First ADX value is the average of the buffer
                    let dx_sum: f64 = dx_buffer.iter().sum();
                    adx_result[i] = dx_sum / period as f64;
                } else if !adx_result[i - 1].is_nan() {
                    // Subsequent ADX values are smoothed
                    adx_result[i] = (adx_result[i - 1] * (period - 1) as f64 + current_dx) / period as f64;
                }
            }
        }

        Ok(PyArray1::from_array(py, &adx_result).to_owned())
    })
}

/// Calculate Alligator indicator (Jaw, Teeth, Lips) - Optimized version
#[pyfunction]
pub fn alligator(source: PyReadonlyArray1<f64>) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let source_array = source.as_array();
        let n = source_array.len();
        
        // Initialize result arrays
        let mut jaw = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 13 {  // Need at least 13 periods for jaw (longest period)
            return Ok((
                PyArray1::from_array(py, &jaw).to_owned(),
                PyArray1::from_array(py, &teeth).to_owned(),
                PyArray1::from_array(py, &lips).to_owned()
            ));
        }
        
        // SMMA parameters
        let jaw_period = 13;
        let teeth_period = 8;
        let lips_period = 5;
        
        let jaw_alpha = 1.0 / jaw_period as f64;
        let teeth_alpha = 1.0 / teeth_period as f64;
        let lips_alpha = 1.0 / lips_period as f64;
        
        // Calculate initial SMA values for each line
        let mut jaw_sum = 0.0;
        let mut teeth_sum = 0.0;
        let mut lips_sum = 0.0;
        
        // Accumulate sums for initial values
        for i in 0..jaw_period {
            jaw_sum += source_array[i];
            if i < teeth_period {
                teeth_sum += source_array[i];
            }
            if i < lips_period {
                lips_sum += source_array[i];
            }
        }
        
        // Initialize SMMA values (without shifts)
        let mut jaw_smma = jaw_sum / jaw_period as f64;
        let mut teeth_smma = teeth_sum / teeth_period as f64;
        let mut lips_smma = lips_sum / lips_period as f64;
        
        // Calculate all SMMA values first, then apply shifts
        let mut jaw_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut teeth_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        let mut lips_unshifted = Array1::<f64>::from_elem(n, f64::NAN);
        
        // Set initial SMMA values
        jaw_unshifted[jaw_period - 1] = jaw_smma;
        teeth_unshifted[teeth_period - 1] = teeth_smma;
        lips_unshifted[lips_period - 1] = lips_smma;
        
        // Calculate subsequent SMMA values using single pass
        for i in jaw_period..n {
            // Update jaw (13-period SMMA)
            jaw_smma = jaw_alpha * source_array[i] + (1.0 - jaw_alpha) * jaw_smma;
            jaw_unshifted[i] = jaw_smma;
            
            // Update teeth (8-period SMMA) if we have enough data
            if i >= teeth_period {
                teeth_smma = teeth_alpha * source_array[i] + (1.0 - teeth_alpha) * teeth_smma;
                teeth_unshifted[i] = teeth_smma;
            }
            
            // Update lips (5-period SMMA) if we have enough data
            if i >= lips_period {
                lips_smma = lips_alpha * source_array[i] + (1.0 - lips_alpha) * lips_smma;
                lips_unshifted[i] = lips_smma;
            }
        }
        
        // Apply shifts inline (forward shifts)
        // Jaw: shift by 8 periods forward
        for i in 0..(n - 8) {
            if i + 8 < n && !jaw_unshifted[i].is_nan() {
                jaw[i + 8] = jaw_unshifted[i];
            }
        }
        
        // Teeth: shift by 5 periods forward
        for i in 0..(n - 5) {
            if i + 5 < n && !teeth_unshifted[i].is_nan() {
                teeth[i + 5] = teeth_unshifted[i];
            }
        }
        
        // Lips: shift by 3 periods forward
        for i in 0..(n - 3) {
            if i + 3 < n && !lips_unshifted[i].is_nan() {
                lips[i + 3] = lips_unshifted[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &jaw).to_owned(),
            PyArray1::from_array(py, &teeth).to_owned(),
            PyArray1::from_array(py, &lips).to_owned()
        ))
    })
} 

/// Calculate DI (Directional Indicator) - Optimized version
#[pyfunction]
pub fn di(candles: PyReadonlyArray2<f64>, period: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        // Initialize result arrays
        let mut plus_di = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_di = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n < 2 || n < period + 1 {
            return Ok((
                PyArray1::from_array(py, &plus_di).to_owned(),
                PyArray1::from_array(py, &minus_di).to_owned()
            ));
        }
        
        // Extract OHLCV data (assuming standard format: open, high, low, close, volume)
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        let close = candles_array.column(2);
        
        // Initialize smoothed values for Wilder's smoothing
        let mut tr_smooth = 0.0;
        let mut plus_dm_smooth = 0.0;
        let mut minus_dm_smooth = 0.0;
        
        // Calculate initial sums for the first 'period' values
        for i in 1..=period {
            // Calculate True Range (TR)
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            // Calculate Directional Movements (+DM and -DM)
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Accumulate for initial smoothed values
            tr_smooth += current_tr;
            plus_dm_smooth += current_plus_dm;
            minus_dm_smooth += current_minus_dm;
        }
        
        // Calculate first DI values at index 'period'
        if tr_smooth > 0.0 {
            plus_di[period] = 100.0 * plus_dm_smooth / tr_smooth;
            minus_di[period] = 100.0 * minus_dm_smooth / tr_smooth;
        } else {
            plus_di[period] = 0.0;
            minus_di[period] = 0.0;
        }
        
        // Calculate subsequent DI values using Wilder's smoothing
        for i in (period + 1)..n {
            // Calculate current TR, +DM, -DM
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            let current_tr = hl.max(hc).max(lc);
            
            let h_diff = high[i] - high[i - 1];
            let l_diff = low[i - 1] - low[i];
            
            let current_plus_dm = if h_diff > l_diff && h_diff > 0.0 { h_diff } else { 0.0 };
            let current_minus_dm = if l_diff > h_diff && l_diff > 0.0 { l_diff } else { 0.0 };
            
            // Apply Wilder's smoothing: smoothed[i] = (smoothed[i-1] * (period - 1) + current) / period
            tr_smooth = (tr_smooth * (period - 1) as f64 + current_tr) / period as f64;
            plus_dm_smooth = (plus_dm_smooth * (period - 1) as f64 + current_plus_dm) / period as f64;
            minus_dm_smooth = (minus_dm_smooth * (period - 1) as f64 + current_minus_dm) / period as f64;
            
            // Calculate DI values
            if tr_smooth > 0.0 {
                plus_di[i] = 100.0 * plus_dm_smooth / tr_smooth;
                minus_di[i] = 100.0 * minus_dm_smooth / tr_smooth;
            } else {
                plus_di[i] = 0.0;
                minus_di[i] = 0.0;
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_di).to_owned(),
            PyArray1::from_array(py, &minus_di).to_owned()
        ))
    })
}

/// Calculate Directional Movement (DM) - Ultra-optimized version
#[pyfunction]
pub fn dm(candles: PyReadonlyArray2<f64>, period: usize) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.nrows();
        
        let mut plus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        let mut minus_dm = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &plus_dm).to_owned(),
                PyArray1::from_array(py, &minus_dm).to_owned()
            ));
        }
        
        // Extract price data
        let high = candles_array.column(3);
        let low = candles_array.column(4);
        
        // Calculate raw directional movements
        let mut raw_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut raw_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];
            
            if up_move > down_move && up_move > 0.0 {
                raw_plus[i] = up_move;
            } else {
                raw_plus[i] = 0.0;
            }
            
            if down_move > up_move && down_move > 0.0 {
                raw_minus[i] = down_move;
            } else {
                raw_minus[i] = 0.0;
            }
        }
        
        // Apply Wilder's smoothing
        if n > period {
            // Calculate initial sum
            let mut sum_plus = 0.0;
            let mut sum_minus = 0.0;
            
            for i in 1..=period {
                sum_plus += raw_plus[i];
                sum_minus += raw_minus[i];
            }
            
            plus_dm[period] = sum_plus;
            minus_dm[period] = sum_minus;
            
            // Apply Wilder's smoothing formula
            for i in (period + 1)..n {
                plus_dm[i] = plus_dm[i - 1] - (plus_dm[i - 1] / period as f64) + raw_plus[i];
                minus_dm[i] = minus_dm[i - 1] - (minus_dm[i - 1] / period as f64) + raw_minus[i];
            }
        }
        
        Ok((
            PyArray1::from_array(py, &plus_dm).to_owned(),
            PyArray1::from_array(py, &minus_dm).to_owned()
        ))
    })
}

/// Calculate DX (Directional Movement Index) - matches the reference Python implementation exactly
/// Uses the reference rma seeding behavior (newseries[-1] = source[-1])
#[pyfunction]
pub fn dx(
    candles: PyReadonlyArray2<f64>,
    di_length: usize,
    adx_smoothing: usize,
    _sequential: bool,
) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];

        if n == 0 {
            let empty = Array1::<f64>::from_elem(0, f64::NAN);
            return Ok((
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
                PyArray1::from_array(py, &empty).to_owned(),
            ));
        }

        let high = candles_array.slice(s![.., 3]);
        let low = candles_array.slice(s![.., 4]);
        let close = candles_array.slice(s![.., 2]);

        // Build plusDM, minusDM, true_range arrays (matches _fast_dm_tr)
        let mut plus_dm = vec![0.0f64; n];
        let mut minus_dm = vec![0.0f64; n];
        let mut tr = vec![0.0f64; n];
        tr[0] = high[0usize] - low[0usize];
        for i in 1..n {
            let up = high[i] - high[i - 1];
            let down = low[i - 1] - low[i];
            if up > down && up > 0.0 { plus_dm[i] = up; }
            if down > up && down > 0.0 { minus_dm[i] = down; }
            let a = high[i] - low[i];
            let b = (high[i] - close[i - 1]).abs();
            let c = (low[i] - close[i - 1]).abs();
            tr[i] = a.max(b).max(c);
        }

        // reference rma: r[0] = alpha * src[0] + (1-alpha) * src[n-1], then EMA
        let alpha_di = 1.0 / di_length as f64;
        let one_minus_di = 1.0 - alpha_di;
        let mut tr_rma = vec![0.0f64; n];
        let mut plus_rma = vec![0.0f64; n];
        let mut minus_rma = vec![0.0f64; n];
        tr_rma[0] = alpha_di * tr[0] + one_minus_di * tr[n - 1];
        plus_rma[0] = alpha_di * plus_dm[0] + one_minus_di * plus_dm[n - 1];
        minus_rma[0] = alpha_di * minus_dm[0] + one_minus_di * minus_dm[n - 1];
        for i in 1..n {
            tr_rma[i] = alpha_di * tr[i] + one_minus_di * tr_rma[i - 1];
            plus_rma[i] = alpha_di * plus_dm[i] + one_minus_di * plus_rma[i - 1];
            minus_rma[i] = alpha_di * minus_dm[i] + one_minus_di * minus_rma[i - 1];
        }

        // plusDI, minusDI, directional_index
        let mut plus_di = vec![0.0f64; n];
        let mut minus_di = vec![0.0f64; n];
        let mut di_idx = vec![0.0f64; n];
        for i in 0..n {
            if tr_rma[i] != 0.0 {
                plus_di[i] = 100.0 * plus_rma[i] / tr_rma[i];
                minus_di[i] = 100.0 * minus_rma[i] / tr_rma[i];
            }
            let di_sum = plus_di[i] + minus_di[i];
            let di_diff = (plus_di[i] - minus_di[i]).abs();
            di_idx[i] = if di_sum == 0.0 { di_diff } else { di_diff / di_sum };
        }

        // adx = 100 * rma(directional_index, adx_smoothing) — same reference seeding
        let alpha_adx = 1.0 / adx_smoothing as f64;
        let one_minus_adx = 1.0 - alpha_adx;
        let mut adx = vec![0.0f64; n];
        adx[0] = 100.0 * (alpha_adx * di_idx[0] + one_minus_adx * di_idx[n - 1]);
        for i in 1..n {
            adx[i] = one_minus_adx * adx[i - 1] + 100.0 * alpha_adx * di_idx[i];
        }

        Ok((
            PyArray1::from_vec(py, adx).to_owned(),
            PyArray1::from_vec(py, plus_di).to_owned(),
            PyArray1::from_vec(py, minus_di).to_owned(),
        ))
    })
}

/// Calculate VI (Vortex Indicator)
#[pyfunction] 
pub fn vi(
    candles: PyReadonlyArray2<f64>,
    period: usize,
    sequential: bool
) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let candles_array = candles.as_array();
        let n = candles_array.shape()[0];
        let mut vi_plus = Array1::<f64>::from_elem(n, f64::NAN);
        let mut vi_minus = Array1::<f64>::from_elem(n, f64::NAN);
        
        if n <= period {
            return Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ));
        }
        
        let close = candles_array.slice(s![.., 2]).to_owned();
        let high = candles_array.slice(s![.., 3]).to_owned();
        let low = candles_array.slice(s![.., 4]).to_owned();
        
        // Calculate True Range, VP and VM for each period
        let mut tr = Array1::<f64>::zeros(n);
        let mut vp = Array1::<f64>::zeros(n);
        let mut vm = Array1::<f64>::zeros(n);
        
        // First candle
        if n > 0 {
            tr[0usize] = high[0usize] - low[0usize];
        }
        
        // Calculate TR, VP, VM for each candle
        for i in 1..n {
            let hl = high[i] - low[i];
            let hpc = (high[i] - close[i - 1]).abs();
            let lpc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hpc).max(lpc);
            
            vp[i] = (high[i] - low[i - 1]).abs();
            vm[i] = (low[i] - high[i - 1]).abs();
        }
        
        // Calculate rolling sums and VI values
        for i in period..n {
            let start_idx = i + 1 - period;
            
            let sum_tr: f64 = tr.slice(s![start_idx..=i]).sum();
            let sum_vp: f64 = vp.slice(s![start_idx..=i]).sum();
            let sum_vm: f64 = vm.slice(s![start_idx..=i]).sum();
            
            if sum_tr != 0.0 {
                vi_plus[i] = sum_vp / sum_tr;
                vi_minus[i] = sum_vm / sum_tr;
            }
        }
        
        if sequential {
            Ok((
                PyArray1::from_array(py, &vi_plus).to_owned(),
                PyArray1::from_array(py, &vi_minus).to_owned()
            ))
        } else {
            let plus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_plus[n-1] } else { f64::NAN });
            let minus_result = Array1::<f64>::from_elem(1, if n > 0 { vi_minus[n-1] } else { f64::NAN });
            Ok((
                PyArray1::from_array(py, &plus_result).to_owned(),
                PyArray1::from_array(py, &minus_result).to_owned()
            ))
        }
    })
}

/// ADXR — Average Directional Movement Index Rating
#[pyfunction]
pub fn adxr(candles: PyReadonlyArray2<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut tr = vec![0.0f64; n];
        let mut dmp = vec![0.0f64; n];
        let mut dmm = vec![0.0f64; n];
        tr[0] = c[[0,3]] - c[[0,4]];
        for i in 1..n {
            tr[i] = (c[[i,3]] - c[[i,4]]).max((c[[i,3]] - c[[i-1,2]]).abs()).max((c[[i,4]] - c[[i-1,2]]).abs());
            let up = c[[i,3]] - c[[i-1,3]];
            let dn = c[[i-1,4]] - c[[i,4]];
            dmp[i] = if up > dn && up > 0.0 { up } else { 0.0 };
            dmm[i] = if dn > up && dn > 0.0 { dn } else { 0.0 };
        }
        let mut str_ = vec![0.0f64; n];
        let mut sdmp = vec![0.0f64; n];
        let mut sdmm = vec![0.0f64; n];
        str_[0] = tr[0]; sdmp[0] = dmp[0]; sdmm[0] = dmm[0];
        for i in 1..n {
            str_[i] = str_[i-1] - str_[i-1] / period as f64 + tr[i];
            sdmp[i] = sdmp[i-1] - sdmp[i-1] / period as f64 + dmp[i];
            sdmm[i] = sdmm[i-1] - sdmm[i-1] / period as f64 + dmm[i];
        }
        let mut dx = vec![0.0f64; n];
        for i in 0..n {
            if str_[i] != 0.0 {
                let di_p = sdmp[i] / str_[i] * 100.0;
                let di_m = sdmm[i] / str_[i] * 100.0;
                let s = di_p + di_m;
                dx[i] = if s != 0.0 { (di_p - di_m).abs() / s * 100.0 } else { 0.0 };
            }
        }
        let mut adx_val = vec![f64::NAN; n];
        for i in (period-1)..n {
            adx_val[i] = dx[i+1-period..=i].iter().sum::<f64>() / period as f64;
        }
        let mut result = vec![f64::NAN; n];
        for i in period..n {
            if !adx_val[i].is_nan() && !adx_val[i-period].is_nan() {
                result[i] = (adx_val[i] + adx_val[i-period]) / 2.0;
            }
        }
        Ok(PyArray1::from_vec(py, result).to_owned())
    })
}

/// SuperTrend → (trend, changed)
#[pyfunction]
pub fn supertrend(candles: PyReadonlyArray2<f64>, period: usize, factor: f64) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let h: Vec<f64> = (0..n).map(|i| c[[i,3]]).collect();
        let l: Vec<f64> = (0..n).map(|i| c[[i,4]]).collect();
        let cl: Vec<f64> = (0..n).map(|i| c[[i,2]]).collect();
        let atr_vals = ih_atr_wilder(&h, &l, &cl, period);
        let mut upper_basic = vec![0.0f64; n];
        let mut lower_basic = vec![0.0f64; n];
        let mut upper_band = vec![0.0f64; n];
        let mut lower_band = vec![0.0f64; n];
        for i in 0..n {
            let mid = (c[[i,3]] + c[[i,4]]) / 2.0;
            let atr = if atr_vals[i].is_nan() { 0.0 } else { atr_vals[i] };
            upper_basic[i] = mid + factor * atr;
            lower_basic[i] = mid - factor * atr;
            upper_band[i] = upper_basic[i];
            lower_band[i] = lower_basic[i];
        }
        let mut trend = vec![0.0f64; n];
        let mut changed = vec![0.0f64; n];
        let idx = period.saturating_sub(1);
        if idx < n {
            trend[idx] = if c[[idx,2]] <= upper_band[idx] { upper_band[idx] } else { lower_band[idx] };
        }
        for i in period..n {
            let p = i - 1;
            let prev_cl = c[[p,2]];
            upper_band[i] = if prev_cl <= upper_band[p] {
                upper_basic[i].min(upper_band[p])
            } else { upper_basic[i] };
            lower_band[i] = if prev_cl >= lower_band[p] {
                lower_basic[i].max(lower_band[p])
            } else { lower_basic[i] };
            if trend[p] == upper_band[p] {
                if c[[i,2]] <= upper_band[i] {
                    trend[i] = upper_band[i]; changed[i] = 0.0;
                } else {
                    trend[i] = lower_band[i]; changed[i] = 1.0;
                }
            } else if c[[i,2]] >= lower_band[i] {
                trend[i] = lower_band[i]; changed[i] = 0.0;
            } else {
                trend[i] = upper_band[i]; changed[i] = 1.0;
            }
        }
        Ok((
            PyArray1::from_vec(py, trend).to_owned(),
            PyArray1::from_vec(py, changed).to_owned(),
        ))
    })
}

/// Parabolic SAR
#[pyfunction]
pub fn sar(candles: PyReadonlyArray2<f64>, acceleration: f64, maximum: f64) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        if n == 0 { return Ok(PyArray1::from_vec(py, vec![]).to_owned()); }
        let high: Vec<f64> = (0..n).map(|i| c[[i,3]]).collect();
        let low: Vec<f64> = (0..n).map(|i| c[[i,4]]).collect();
        let mut sar_v = vec![0.0f64; n];
        let mut uptrend = high[1] > high[0];
        sar_v[0] = if uptrend { low[0] } else { high[0] };
        let mut ep = if uptrend { high[0] } else { low[0] };
        let mut af = acceleration;
        for i in 1..n {
            let prev_sar = sar_v[i-1];
            let mut sar_temp = if uptrend {
                let mut s = prev_sar + af * (ep - prev_sar);
                if i >= 2 { s = s.min(low[i-1]).min(low[i-2]); } else { s = s.min(low[i-1]); }
                s
            } else {
                let mut s = prev_sar - af * (prev_sar - ep);
                if i >= 2 { s = s.max(high[i-1]).max(high[i-2]); } else { s = s.max(high[i-1]); }
                s
            };
            if uptrend {
                if low[i] < sar_temp {
                    sar_temp = ep; uptrend = false; af = acceleration; ep = low[i];
                } else if high[i] > ep {
                    ep = high[i]; af = (af + acceleration).min(maximum);
                }
            } else if high[i] > sar_temp {
                sar_temp = ep; uptrend = true; af = acceleration; ep = high[i];
            } else if low[i] < ep {
                ep = low[i]; af = (af + acceleration).min(maximum);
            }
            sar_v[i] = sar_temp;
        }
        Ok(PyArray1::from_vec(py, sar_v).to_owned())
    })
}

/// SafeZone Stops
#[pyfunction]
pub fn safezonestop(candles: PyReadonlyArray2<f64>, period: usize, mult: f64, max_lookback: usize, direction: bool) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let c = candles.as_array();
        let n = c.shape()[0];
        let mut diff_high = vec![0.0f64; n];
        let mut diff_low = vec![0.0f64; n];
        for i in 1..n {
            let dh = c[[i,3]] - c[[i-1,3]];
            let dl = c[[i-1,4]] - c[[i,4]];
            diff_high[i] = if dh.is_nan() { 0.0 } else { dh };
            diff_low[i] = if dl.is_nan() { 0.0 } else { dl };
        }
        let mut raw_plus = vec![0.0f64; n];
        let mut raw_minus = vec![0.0f64; n];
        for i in 0..n {
            raw_plus[i] = if diff_high[i] > diff_low[i] && diff_high[i] > 0.0 { diff_high[i] } else { 0.0 };
            raw_minus[i] = if diff_low[i] > diff_high[i] && diff_low[i] > 0.0 { diff_low[i] } else { 0.0 };
        }
        // Wilder smoothing: smoothed[i] = alpha * smoothed[i-1] + raw[i], alpha = (period-1)/period
        let alpha = 1.0 - 1.0 / period as f64;
        let mut plus_dm = vec![0.0f64; n];
        let mut minus_dm = vec![0.0f64; n];
        plus_dm[0] = raw_plus[0]; minus_dm[0] = raw_minus[0];
        for i in 1..n {
            plus_dm[i] = alpha * plus_dm[i-1] + raw_plus[i];
            minus_dm[i] = alpha * minus_dm[i-1] + raw_minus[i];
        }
        let mut intermediate = vec![0.0f64; n];
        for i in 1..n {
            intermediate[i] = if direction {
                c[[i-1,4]] - mult * minus_dm[i]  // long: last_low - mult * minus_dm
            } else {
                c[[i-1,3]] + mult * plus_dm[i]   // short: last_high + mult * plus_dm
            };
        }
        let mut r = vec![0.0f64; n];
        for i in 0..n {
            let start = i.saturating_sub(max_lookback - 1);
            r[i] = if direction {
                intermediate[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                intermediate[start..=i].iter().cloned().fold(f64::INFINITY, f64::min)
            };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}
