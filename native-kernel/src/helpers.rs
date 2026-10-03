//! Inline helpers shared across indicator modules.
//!
//! These are not exposed to Python; they exist to keep indicator
//! implementations short and to avoid recomputing primitive series
//! (EMA, SMA, WMA, Ehlers' 2-pole supersmoother and 1-pole high-pass,
//! Wilder ATR, rolling population std). A handful of `ndarray::Array1`
//! variants are also defined for indicators that operate on Array1
//! intermediate results (`stoch`, `stochf`, `wt`).

use ndarray::{s, Array1};

/// Standard EMA with alpha = 2 / (period + 1) seeded by the first sample.
pub(crate) fn ih_ema(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![0.0f64; n];
    if n == 0 {
        return r;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    r[0] = source[0];
    for i in 1..n {
        r[i] = alpha * source[i] + (1.0 - alpha) * r[i - 1];
    }
    r
}

/// Simple moving average; result[0..period-1] is NaN.
pub(crate) fn ih_sma(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![f64::NAN; n];
    if period == 0 || n < period {
        return r;
    }
    let pf = period as f64;
    let mut sum: f64 = source[..period].iter().sum();
    r[period - 1] = sum / pf;
    for i in period..n {
        sum += source[i] - source[i - period];
        r[i] = sum / pf;
    }
    r
}

/// Weighted moving average with linearly increasing weights.
pub(crate) fn ih_wma(source: &[f64], period: usize) -> Vec<f64> {
    let n = source.len();
    let mut r = vec![0.0f64; n];
    if period == 0 {
        return r;
    }
    let ws: f64 = (1..=period).map(|x| x as f64).sum();
    for i in period.saturating_sub(1)..n {
        if i + 1 < period {
            continue;
        }
        let mut w = 0.0;
        for j in 0..period {
            w += source[i - j] * (period - j) as f64;
        }
        r[i] = w / ws;
    }
    r
}

/// Ehlers 2-pole supersmoother.
pub(crate) fn ih_supersmoother_2pole(source: &[f64], period: f64) -> Vec<f64> {
    let n = source.len();
    let mut r = source.to_vec();
    let pi = std::f64::consts::PI;
    let a = (-1.414 * pi / period).exp();
    let b = 2.0 * a * (1.414 * pi / period).cos();
    let c1 = (1.0 + a * a - b) / 2.0;
    for i in 2..n {
        r[i] = c1 * (source[i] + source[i - 1]) + b * r[i - 1] - a * a * r[i - 2];
    }
    r
}

/// Ehlers 1-pole high-pass filter.
pub(crate) fn ih_high_pass_1pole(source: &[f64], period: f64) -> Vec<f64> {
    let n = source.len();
    let mut r = source.to_vec();
    let pi = std::f64::consts::PI;
    let sv = (2.0 * pi / period).sin();
    let cv = (2.0 * pi / period).cos();
    let alpha = 1.0 + (sv - 1.0) / cv;
    let c = 1.0 - alpha / 2.0;
    for i in 1..n {
        r[i] = c * source[i] - c * source[i - 1] + (1.0 - alpha) * r[i - 1];
    }
    r
}

/// Wilder ATR: SMA of true range over the first `period` bars, then EMA with alpha = 1/period.
pub(crate) fn ih_atr_wilder(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        tr[i] = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
    }
    let mut r = vec![f64::NAN; n];
    if n < period {
        return r;
    }
    r[period - 1] = tr[..period].iter().sum::<f64>() / period as f64;
    for i in period..n {
        r[i] = (r[i - 1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }
    r
}

/// Rolling population standard deviation over a fixed window.
/// `result[i]` for `i < window - 1` is left as 0.0.
pub(crate) fn rolling_std_pop(source: &[f64], window: usize) -> Vec<f64> {
    let n = source.len();
    let mut result = vec![0.0f64; n];
    if window == 0 || window > n {
        return result;
    }
    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for &v in &source[..window] {
        sum += v;
        sum_sq += v * v;
    }
    let w = window as f64;
    let mean = sum / w;
    let var = (sum_sq / w - mean * mean).max(0.0);
    result[window - 1] = var.sqrt();
    for i in window..n {
        sum += source[i] - source[i - window];
        sum_sq += source[i] * source[i] - source[i - window] * source[i - window];
        let mean = sum / w;
        let var = (sum_sq / w - mean * mean).max(0.0);
        result[i] = var.sqrt();
    }
    result
}

// ============================================================
// ndarray::Array1 variants
// ============================================================

/// SMA over an `Array1`, skipping NaNs inside each window (returns NaN-padded prefix).
/// Used by `stoch` / `stochf` for slow-K and slow-D smoothing.
pub(crate) fn sma_array(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::from_elem(n, f64::NAN);
    if n < period {
        return result;
    }
    for i in (period - 1)..n {
        let start_idx = i + 1 - period;
        let end_idx = i + 1;
        let window = source.slice(s![start_idx..end_idx]);
        let mut sum = 0.0;
        let mut count = 0;
        for &val in window.iter() {
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }
    result
}

/// EMA over an `Array1`, seeded by the first sample. Used by `wt` (wavetrend).
pub(crate) fn ema_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    if n == 0 {
        return result;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = source[0];
    for i in 1..n {
        result[i] = alpha * source[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

/// SMA over an `Array1` with partial windows for the first `period` samples.
/// Used by `wt` (wavetrend) for its trailing smoother.
pub(crate) fn sma_for_wt(source: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = source.len();
    let mut result = Array1::<f64>::zeros(n);
    if n == 0 || period == 0 {
        return result;
    }
    let mut cumsum = 0.0;
    for i in 0..period.min(n) {
        cumsum += source[i];
        result[i] = cumsum / (i as f64 + 1.0);
    }
    if n <= period {
        return result;
    }
    for i in period..n {
        result[i] = result[i - 1] + (source[i] - source[i - period]) / period as f64;
    }
    result
}

/// Truncate a candle timestamp (milliseconds) to a coarser bucket key. Used by `vwap`.
pub(crate) fn get_period_from_timestamp(timestamp: f64, anchor: &str) -> i64 {
    let seconds = (timestamp / 1000.0) as i64;
    match anchor.to_uppercase().as_str() {
        "D" => seconds / 86400,         // daily
        "H" => seconds / 3600,          // hourly
        "M" => seconds / 60,            // minute
        "4H" => seconds / 14400,        // 4-hour
        "12H" => seconds / 43200,       // 12-hour
        "W" => seconds / 604800,        // weekly
        "MN" => seconds / 2592000,      // monthly (~30 days)
        _ => seconds / 86400,           // default daily
    }
}
