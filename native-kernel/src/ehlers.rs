//! Filters from John Ehlers' family (super-smoother, high-pass, MAMA, reflex, voss, …).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::types::{PyArrTuple2, PyArrTuple3, PyArrTuple4};

use crate::helpers::{ih_high_pass_1pole, ih_supersmoother_2pole};

/// SuperSmoother 2-pole
#[pyfunction]
pub fn supersmoother(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let r = ih_supersmoother_2pole(&src, period as f64);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// SuperSmoother 3-pole
#[pyfunction]
pub fn supersmoother_3_pole(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let a = (-pi / period as f64).exp();
        let b = 2.0 * a * (1.738 * pi / period as f64).cos();
        let c = a * a;
        let mut r = src.to_vec();
        for i in 3..n {
            r[i] = (1.0 - c * c - b + b * c) * src[i]
                 + (b + c) * r[i-1]
                 + (-c - b * c) * r[i-2]
                 + c * c * r[i-3];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// High Pass 1-pole
#[pyfunction]
pub fn high_pass(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let r = ih_high_pass_1pole(&src, period as f64);
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// High Pass 2-pole
#[pyfunction]
pub fn high_pass_2_pole(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let k = 0.707f64;
        let sv = (2.0 * pi * k / period as f64).sin();
        let cv = (2.0 * pi * k / period as f64).cos();
        let alpha = 1.0 + (sv - 1.0) / cv;
        let c = (1.0 - alpha / 2.0).powi(2);
        let mut r = src.to_vec();
        for i in 2..n {
            r[i] = c * src[i]
                 - 2.0 * c * src[i-1]
                 + c * src[i-2]
                 + 2.0 * (1.0 - alpha) * r[i-1]
                 - (1.0 - alpha).powi(2) * r[i-2];
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Bandpass Filter → (bp, bp_normalized, signal, trigger)
#[pyfunction]
pub fn bandpass(source: PyReadonlyArray1<f64>, period: usize, bandwidth: f64) -> PyArrTuple4 {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let hp = ih_high_pass_1pole(&src, 4.0 * period as f64 / bandwidth);
        let beta = (2.0 * pi / period as f64).cos();
        let gamma = (2.0 * pi * bandwidth / period as f64).cos();
        let alpha = 1.0 / gamma - (1.0 / (gamma * gamma) - 1.0).sqrt();
        let mut bp = hp.clone();
        for i in 2..n {
            bp[i] = 0.5 * (1.0 - alpha) * (hp[i] - hp[i-2])
                  + beta * (1.0 + alpha) * bp[i-1]
                  - alpha * bp[i-2];
        }
        // AGC
        let k = 0.991f64;
        let mut peak = bp.clone();
        for i in 1..n {
            peak[i] = peak[i-1] * k;
            if bp[i].abs() > peak[i] { peak[i] = bp[i].abs(); }
        }
        let bp_norm: Vec<f64> = (0..n).map(|i| if peak[i] != 0.0 { bp[i] / peak[i] } else { 0.0 }).collect();
        let trigger = ih_high_pass_1pole(&bp_norm, period as f64 / bandwidth / 1.5);
        let signal: Vec<f64> = (0..n).map(|i| {
            if bp_norm[i] < trigger[i] { 1.0 } else if trigger[i] < bp_norm[i] { -1.0 } else { 0.0 }
        }).collect();
        Ok((
            PyArray1::from_vec(py, bp).to_owned(),
            PyArray1::from_vec(py, bp_norm).to_owned(),
            PyArray1::from_vec(py, signal).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}

/// Gaussian Filter
#[pyfunction]
pub fn gauss(source: PyReadonlyArray1<f64>, period: usize, poles: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src_arr = source.as_array();
        let n_total = src_arr.len();
        let src: Vec<f64> = src_arr.iter().filter(|x| !x.is_nan()).cloned().collect();
        let to_fill = n_total - src.len();
        let pi = std::f64::consts::PI;
        let beta = (1.0 - (2.0 * pi / period as f64).cos()) / (2.0f64.powf(1.0 / poles as f64) - 1.0);
        let alpha = -beta + (beta * beta + 2.0 * beta).sqrt();
        let m = src.len();
        let mut fil = vec![0.0f64; poles + m];
        let coeff: Vec<f64> = match poles {
            1 => vec![alpha, 1.0 - alpha],
            2 => vec![alpha.powi(2), 2.0 * (1.0 - alpha), -(1.0 - alpha).powi(2)],
            3 => vec![alpha.powi(3), 3.0*(1.0-alpha), -3.0*(1.0-alpha).powi(2), (1.0-alpha).powi(3)],
            _ => vec![alpha.powi(4), 4.0*(1.0-alpha), -6.0*(1.0-alpha).powi(2), 4.0*(1.0-alpha).powi(3), -(1.0-alpha).powi(4)],
        };
        for i in 0..m {
            let val: f64 = coeff[0] * src[i] + coeff[1..].iter().enumerate().map(|(k, &c)| {
                let idx = poles + i - 1 - k;
                c * fil[idx]
            }).sum::<f64>();
            fil[poles + i] = val;
        }
        let fil_slice = &fil[poles..];
        let mut r = vec![f64::NAN; n_total];
        let start = if to_fill > 0 { to_fill } else { 0 };
        for (k, &v) in fil_slice.iter().enumerate() {
            if start + k < n_total { r[start + k] = v; }
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Reflex indicator
#[pyfunction]
pub fn reflex(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let ssf = ih_supersmoother_2pole(&src, period as f64 / 2.0);
        let n = ssf.len();
        let mut rf = vec![0.0f64; n];
        let mut ms = vec![0.0f64; n];
        let mut sums = vec![0.0f64; n];
        for i in period..n {
            let slope = (ssf[i-period] - ssf[i]) / period as f64;
            let mut my_sum = 0.0f64;
            for t in 1..=period {
                my_sum += (ssf[i] + t as f64 * slope) - ssf[i-t];
            }
            my_sum /= period as f64;
            sums[i] = my_sum;
            ms[i] = 0.04 * sums[i] * sums[i] + 0.96 * ms[i-1];
            if ms[i] > 0.0 { rf[i] = sums[i] / ms[i].sqrt(); }
        }
        Ok(PyArray1::from_vec(py, rf).to_owned())
    })
}

/// TrendFlex indicator
#[pyfunction]
pub fn trendflex(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src: Vec<f64> = source.as_array().to_vec();
        let ssf = ih_supersmoother_2pole(&src, period as f64 / 2.0);
        let n = ssf.len();
        let mut tf = vec![0.0f64; n];
        let mut ms = vec![0.0f64; n];
        let mut sums = vec![0.0f64; n];
        for i in period..n {
            let mut my_sum = 0.0f64;
            for t in 1..=period { my_sum += ssf[i] - ssf[i-t]; }
            my_sum /= period as f64;
            sums[i] = my_sum;
            ms[i] = 0.04 * sums[i] * sums[i] + 0.96 * ms[i-1];
            if ms[i] != 0.0 { tf[i] = sums[i] / ms[i].sqrt(); }
        }
        Ok(PyArray1::from_vec(py, tf).to_owned())
    })
}

/// Instantaneous Trendline → (signal, it, trigger)
#[pyfunction]
pub fn itrend(source: PyReadonlyArray1<f64>, alpha: f64) -> PyArrTuple3 {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut it = src.to_vec();
        for i in 2..7.min(n) {
            it[i] = (src[i] + 2.0 * src[i-1] + src[i-2]) / 4.0;
        }
        for i in 7..n {
            it[i] = (alpha - alpha * alpha / 4.0) * src[i]
                  + alpha * alpha / 2.0 * src[i-1]
                  - (alpha - 3.0 * alpha * alpha / 4.0) * src[i-2]
                  + 2.0 * (1.0 - alpha) * it[i-1]
                  - (1.0 - alpha).powi(2) * it[i-2];
        }
        let mut trigger = vec![0.0f64; n];
        let mut signal = vec![0.0f64; n];
        for i in 0..n {
            let lag2 = if i >= 20 { it[i-20] } else { it[i] };
            trigger[i] = 2.0 * it[i] - lag2;
            signal[i] = if trigger[i] > it[i] { 1.0 } else if trigger[i] < it[i] { -1.0 } else { 0.0 };
        }
        Ok((
            PyArray1::from_vec(py, signal).to_owned(),
            PyArray1::from_vec(py, it).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}

/// Voss Predictive Filter → (voss, filt)
#[pyfunction]
pub fn voss(source: PyReadonlyArray1<f64>, period: usize, predict: usize, bandwidth: f64) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let order = 3 * predict;
        let f1 = (2.0 * pi / period as f64).cos();
        let g1 = (bandwidth * 2.0 * pi / period as f64).cos();
        let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();
        let mut filt = vec![0.0f64; n];
        let mut voss_v = vec![0.0f64; n];
        for i in 0..n {
            if i > period && i > 5 && i > order {
                filt[i] = 0.5 * (1.0 - s1) * (src[i] - src[i-2])
                        + f1 * (1.0 + s1) * filt[i-1]
                        - s1 * filt[i-2];
            }
        }
        for i in 0..n {
            if i > period && i > 5 && i > order {
                let sumc: f64 = (0..order).map(|count| {
                    (count + 1) as f64 / order as f64 * voss_v[i - (order - count)]
                }).sum();
                voss_v[i] = (3.0 + order as f64) / 2.0 * filt[i] - sumc;
            }
        }
        Ok((
            PyArray1::from_vec(py, voss_v).to_owned(),
            PyArray1::from_vec(py, filt).to_owned(),
        ))
    })
}

/// EDCF — Ehlers Distance Coefficient Filter
#[pyfunction]
pub fn edcf(source: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut r = vec![f64::NAN; n];
        if n < 2 * period || period < 2 { return Ok(PyArray1::from_vec(py, r).to_owned()); }
        // Precompute dist[k] = sum_{lb=1}^{period-1} (src[k] - src[k-lb])^2
        let mut dist = vec![0.0f64; n];
        for k in (period - 1)..n {
            let mut d_sum = 0.0f64;
            for lb in 1..period {
                let d = src[k] - src[k - lb];
                d_sum += d * d;
            }
            dist[k] = d_sum;
        }
        // r[j] uses k = j-i in [j-period+1, j], so rolling window of size `period` ending at j.
        let first_j = 2 * period;
        let mut num: f64 = 0.0;
        let mut coef: f64 = 0.0;
        for k in (first_j - period + 1)..=first_j {
            num += dist[k] * src[k];
            coef += dist[k];
        }
        r[first_j] = if coef != 0.0 { num / coef } else { 0.0 };
        for j in (first_j + 1)..n {
            // Slide window: remove k=j-period, add k=j
            num -= dist[j - period] * src[j - period];
            coef -= dist[j - period];
            num += dist[j] * src[j];
            coef += dist[j];
            r[j] = if coef != 0.0 { num / coef } else { 0.0 };
        }
        Ok(PyArray1::from_vec(py, r).to_owned())
    })
}

/// Correlation Cycle → (real, imag, angle, state)
#[pyfunction]
pub fn correlation_cycle(source: PyReadonlyArray1<f64>, period: usize, threshold: f64) -> PyArrTuple4 {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pix2 = 4.0 * (1.0f64).asin();
        let period = period.max(2);
        let pf = period as f64;
        let mut real_part = vec![f64::NAN; n];
        let mut imag_part = vec![f64::NAN; n];

        // Precompute trig terms (depend only on j and period)
        let yc: Vec<f64> = (0..period).map(|j| (pix2 * (j as f64 + 1.0) / pf).cos()).collect();
        let ys: Vec<f64> = (0..period).map(|j| -((pix2 * (j as f64 + 1.0) / pf).sin())).collect();
        let ry: f64 = yc.iter().sum();
        let iy: f64 = ys.iter().sum();
        let ryy: f64 = yc.iter().map(|v| v * v).sum();
        let iyy: f64 = ys.iter().map(|v| v * v).sum();
        let t2 = pf * ryy - ry * ry;
        let u2 = pf * iyy - iy * iy;
        if !(t2 > 0.0 && u2 > 0.0) {
            // degenerate — fall through with NaN results
        }
        let t2_sqrt = if t2 > 0.0 { t2.sqrt() } else { 0.0 };
        let u2_sqrt = if u2 > 0.0 { u2.sqrt() } else { 0.0 };

        for i in period..n {
            let (mut rx, mut rxx, mut rxy, mut ixy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..period {
                let v = src[i - j - 1];
                let x = if v.is_nan() { 0.0 } else { v };
                rx += x;
                rxx += x * x;
                rxy += x * yc[j];
                ixy += x * ys[j];
            }
            let t1 = pf * rxx - rx * rx;
            if t1 > 0.0 && t2_sqrt > 0.0 {
                real_part[i] = (pf * rxy - rx * ry) / (t1.sqrt() * t2_sqrt);
            }
            if t1 > 0.0 && u2_sqrt > 0.0 {
                imag_part[i] = (pf * ixy - rx * iy) / (t1.sqrt() * u2_sqrt);
            }
        }
        let half_pi = 1.0f64.asin();
        // Initialize to NaN to match Python behavior (imagPart==0 gives 0.0, NaN gives NaN)
        let mut angle = vec![f64::NAN; n];
        for i in 0..n {
            let im = imag_part[i];
            let re = real_part[i];
            if im.is_nan() {
                // stays NaN
            } else if im == 0.0 {
                angle[i] = 0.0;
                // no subtraction since imag > 0 is false
            } else {
                let mut a = ((re / im).atan() + half_pi).to_degrees();
                if im > 0.0 { a -= 180.0; }
                angle[i] = a;
            }
        }
        // Vectorized update: use original angles (clone) to match Python's non-sequential np.where
        let orig_angle = angle.clone();
        for i in 1..n {
            let prior = orig_angle[i-1];
            // NaN > x is false in Rust, so NaN prior means no update
            if prior > angle[i] && prior - angle[i] < 270.0 { angle[i] = prior; }
        }
        let mut state = vec![0.0f64; n];
        for i in 1..n {
            let prior = angle[i-1];
            if (angle[i] - prior).abs() < threshold {
                state[i] = if angle[i] >= 0.0 { 1.0 } else if angle[i] < 0.0 { -1.0 } else { 0.0 };
            }
        }
        Ok((
            PyArray1::from_vec(py, real_part).to_owned(),
            PyArray1::from_vec(py, imag_part).to_owned(),
            PyArray1::from_vec(py, angle).to_owned(),
            PyArray1::from_vec(py, state).to_owned(),
        ))
    })
}

/// MAMA — MESA Adaptive Moving Average → (mama, fama)
#[pyfunction]
pub fn mama(source: PyReadonlyArray1<f64>, fastlimit: f64, slowlimit: f64) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let pi = std::f64::consts::PI;
        let mut sp = vec![0.0f64; n];
        let mut dt = vec![0.0f64; n];
        let mut q1 = vec![0.0f64; n];
        let mut i1 = vec![0.0f64; n];
        let mut ji = vec![0.0f64; n];
        let mut jq = vec![0.0f64; n];
        let mut i2 = vec![0.0f64; n];
        let mut q2 = vec![0.0f64; n];
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        let mut p1 = vec![0.0f64; n];
        let mut p2: f64;
        let mut p3 = vec![0.0f64; n];
        let mut p_arr = vec![0.0f64; n];
        let mut phase = vec![0.0f64; n];
        let mut mama_v = vec![0.0f64; n];
        let mut fama_v = vec![0.0f64; n];
        let g = |i: usize, arr: &Vec<f64>| -> f64 { if i < arr.len() { arr[i] } else { 0.0 } };
        let gs = |i: usize| -> f64 { if i < n { src[i] } else { 0.0 } };
        for i in 1..n {
            sp[i] = (4.0*gs(i) + 3.0*gs(i.saturating_sub(1)) + 2.0*gs(i.saturating_sub(2)) + gs(i.saturating_sub(3))) / 10.0;
            let pv = g(i.saturating_sub(1), &p_arr);
            let c075p = 0.075 * pv + 0.54;
            dt[i] = (0.0962*sp[i] + 0.5769*g(i.saturating_sub(2),&sp) - 0.5769*g(i.saturating_sub(4),&sp) - 0.0962*g(i.saturating_sub(6),&sp)) * c075p;
            q1[i] = (0.0962*dt[i] + 0.5769*g(i.saturating_sub(2),&dt) - 0.5769*g(i.saturating_sub(4),&dt) - 0.0962*g(i.saturating_sub(6),&dt)) * c075p;
            i1[i] = g(i.saturating_sub(3), &dt);
            ji[i] = (0.0962*i1[i] + 0.5769*g(i.saturating_sub(2),&i1) - 0.5769*g(i.saturating_sub(4),&i1) - 0.0962*g(i.saturating_sub(6),&i1)) * c075p;
            jq[i] = (0.0962*q1[i] + 0.5769*g(i.saturating_sub(2),&q1) - 0.5769*g(i.saturating_sub(4),&q1) - 0.0962*g(i.saturating_sub(6),&q1)) * c075p;
            let i2t = i1[i] - jq[i];
            let q2t = q1[i] + ji[i];
            i2[i] = 0.2*i2t + 0.8*g(i.saturating_sub(1),&i2);
            q2[i] = 0.2*q2t + 0.8*g(i.saturating_sub(1),&q2);
            let ret = i2[i]*g(i.saturating_sub(1),&i2) + q2[i]*g(i.saturating_sub(1),&q2);
            let imt = i2[i]*g(i.saturating_sub(1),&q2) - q2[i]*g(i.saturating_sub(1),&i2);
            re[i] = 0.2*ret + 0.8*g(i.saturating_sub(1),&re);
            im[i] = 0.2*imt + 0.8*g(i.saturating_sub(1),&im);
            p1[i] = if im[i] != 0.0 && re[i] != 0.0 { 2.0*pi/(im[i]/re[i]).atan() } else { pv };
            let p1v = p1[i];
            p2 = if p1v > 1.5*pv { 1.5*pv } else if p1v < 0.67*pv { 0.67*pv } else { p1v };
            p3[i] = p2.clamp(6.0, 50.0);
            p_arr[i] = 0.2*p3[i] + 0.8*pv;
            phase[i] = if i1[i] != 0.0 { (q1[i]/i1[i]).atan() * 180.0/pi } else { 0.0 };
            let dphase = (g(i.saturating_sub(1),&phase) - phase[i]).max(1.0);
            let alpha_t = (fastlimit / dphase).max(slowlimit).min(fastlimit);
            mama_v[i] = alpha_t*src[i] + (1.0-alpha_t)*mama_v[i.saturating_sub(1)];
            fama_v[i] = 0.5*alpha_t*mama_v[i] + (1.0-0.5*alpha_t)*fama_v[i.saturating_sub(1)];
        }
        Ok((
            PyArray1::from_vec(py, mama_v).to_owned(),
            PyArray1::from_vec(py, fama_v).to_owned(),
        ))
    })
}

/// PMA — Predictive Moving Average → (predict, trigger)
#[pyfunction]
pub fn pma(source: PyReadonlyArray1<f64>) -> PyArrTuple2 {
    Python::with_gil(|py| {
        let src = source.as_array();
        let n = src.len();
        let mut predict = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];
        let mut wma1 = vec![0.0f64; n];
        for j in 6..n {
            wma1[j] = (7.0*src[j] + 6.0*src[j-1] + 5.0*src[j-2] + 4.0*src[j-3] + 3.0*src[j-4] + 2.0*src[j-5] + src[j-6]) / 28.0;
            let wma2 = (7.0*wma1[j] + 6.0*wma1[j-1] + 5.0*wma1[j-2] + 4.0*wma1[j-3] + 3.0*wma1[j-4] + 2.0*wma1[j-5] + wma1[j-6]) / 28.0;
            predict[j] = 2.0 * wma1[j] - wma2;
        }
        for j in 6..n {
            if !predict[j].is_nan() && j >= 3 {
                let p3 = if predict[j-3].is_nan() { 0.0 } else { predict[j-3] };
                let p2 = if predict[j-2].is_nan() { 0.0 } else { predict[j-2] };
                let p1 = if predict[j-1].is_nan() { 0.0 } else { predict[j-1] };
                trigger[j] = (4.0*predict[j] + 3.0*p1 + 2.0*p2 + p3) / 10.0;
            }
        }
        Ok((
            PyArray1::from_vec(py, predict).to_owned(),
            PyArray1::from_vec(py, trigger).to_owned(),
        ))
    })
}
