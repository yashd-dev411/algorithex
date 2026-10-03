"""
Factor decomposition and risk measurement.

Two questions a return series raises that a PnL number cannot answer: *what
drove it*, and *how bad could it get*.

Factor decomposition
    Regress returns on candidate factors and report each one's contribution.
    With the intercept, the fitted contributions reconstruct the mean return
    exactly, so the decomposition is complete rather than a hand-wave. Risk
    attribution runs the same machinery on volatility contributions.

Risk measurement
    Value at Risk answers "how bad on a normal day"; Expected Shortfall answers
    "how bad on a bad day", which is the one that sizes a position. Reporting
    only the former is the standard way to understate tail risk.

    A Gaussian VaR is reported alongside a Cornish-Fisher expansion, because a
    normal assumption understates the risk of a fat-tailed return series by a
    wide margin — often by a factor of two or more on real crypto data.

Drawdown attribution
    When equity falls, which factor was holding the loss up? This finds the
    worst peak-to-trough period and ranks factor contributions inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    'FactorModel',
    'factor_decomposition',
    'risk_attribution',
    'historical_var',
    'cornish_fisher_var',
    'expected_shortfall',
    'drawdown_attribution',
]


@dataclass
class FactorModel:
    """Result of regressing returns on factors."""

    names: List[str]
    loadings: np.ndarray
    contributions: np.ndarray
    residual_std: float
    r_squared: float
    observations: int

    def as_dict(self) -> Dict[str, float]:
        """Contribution per factor, plus the unexplained residual."""
        result = {name: float(value) for name, value in zip(self.names, self.contributions)}
        result['_residual_std'] = float(self.residual_std)
        return result


def _align(factors: Dict[str, Sequence], returns: Sequence) -> tuple:
    if not factors:
        raise ValueError('no factors supplied')

    names = list(factors)
    columns = [np.asarray(factors[name], dtype=np.float64).reshape(-1) for name in names]
    lengths = {c.size for c in columns}
    if len(lengths) != 1:
        raise ValueError(f'factors have differing lengths: {sorted(lengths)}')

    y = np.asarray(returns, dtype=np.float64).reshape(-1)
    if y.size != columns[0].size:
        raise ValueError(f'{y.size} returns but factors have {columns[0].size} rows')

    matrix = np.column_stack(columns)
    keep = np.all(np.isfinite(matrix), axis=1) & np.isfinite(y)
    return names, matrix[keep], y[keep]


def factor_decomposition(factors: Dict[str, Sequence], returns: Sequence) -> FactorModel:
    """
    Ordinary least squares decomposition of returns onto factors.

    The intercept is included, so the mean contribution of each factor plus the
    intercept reconstructs the sample mean exactly. Contributions therefore sum
    to something meaningful rather than being a ranking of raw correlations.

    :param factors: mapping of factor name to a per-period series
    :param returns: the return series to explain
    """
    names, matrix, y = _align(factors, returns)
    if matrix.shape[0] <= matrix.shape[1] + 1:
        raise ValueError(
            f'need more observations than factors: {matrix.shape[0]} rows, '
            f'{matrix.shape[1]} factors'
        )

    design = np.column_stack([np.ones(matrix.shape[0]), matrix])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    fitted = design @ coefficients
    residuals = y - fitted
    total_variance = float(np.sum((y - y.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residuals ** 2) / total_variance) if total_variance > 0 else 0.0

    return FactorModel(
        names=names,
        loadings=coefficients[1:],
        contributions=coefficients[1:] * matrix.mean(axis=0),
        residual_std=float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0,
        r_squared=r_squared,
        observations=int(matrix.shape[0]),
    )


def risk_attribution(
    factors: Dict[str, Sequence],
    returns: Sequence,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Which factor owns how much of the portfolio's risk.

    Marginal contribution to risk is the factor's weight times its covariance
    with the total, scaled by total volatility. Contributions sum to total
    volatility, so they are directly comparable and add up to a number you can
    check against your own portfolio.

    :param factors: mapping of factor name to a per-period series
    :param returns: the return series being attributed
    :param weights: factor exposures to use. When omitted they are estimated
        by regressing ``returns`` on the factors, which answers a different
        question (what the data implies) rather than the one usually meant
        (what your actual positions risk).
    :return: absolute risk contribution per factor, summing to portfolio volatility
    """
    names, matrix, y = _align(factors, returns)

    if weights is None:
        weights = np.linalg.lstsq(
            np.column_stack([np.ones(matrix.shape[0]), matrix]), y, rcond=None
        )[0][1:]
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != len(names):
            raise ValueError(
                f'got {weights.size} weights for {len(names)} factors'
            )

    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)

    exposures = covariance @ weights
    total = float(np.sqrt(max(weights @ exposures, 0.0)))
    if total <= 0:
        return {name: 0.0 for name in names}

    # Risk contribution is w_i * (Sigma w)_i / sigma_p. Without dividing by
    # sigma_p the contributions sum to the *variance*, not the volatility they
    # are documented to sum to.
    return {name: float(weights[i] * exposures[i] / total) for i, name in enumerate(names)}


def historical_var(returns, confidence: float = 0.95) -> float:
    """
    Empirical quantile of returns, reported as a positive loss figure.

    Empirical rather than parametric on purpose: the assumption that matters is
    that history resembles the future, not that returns are normal.
    """
    values = _clean(returns, 'returns')
    _check_confidence(confidence)
    quantile = float(np.quantile(values, 1.0 - confidence))
    return float(-quantile)


def cornish_fisher_var(returns, confidence: float = 0.95) -> Dict[str, float]:
    """
    Parametric VaR with a Cornish-Fisher skew and kurtosis correction.

    The Gaussian answer assumes zero skew and 3 kurtosis. Real returns have
    negative skew and fat tails, both of which push the true loss further out
    than a normal would suggest, so the correction usually *raises* the number.
    That difference is precisely the tail risk a normal model hides.
    """
    values = _clean(returns, 'returns')
    _check_confidence(confidence)

    from scipy.stats import norm

    std = float(values.std(ddof=1))
    # A constant series leaves std at floating-point noise rather than exactly
    # zero, which yields a meaningless skew and kurtosis. Compare against a
    # scale-relative floor instead.
    scale = max(float(np.abs(values).max()), 1e-12)
    if std <= scale * 1e-12:
        return {'gaussian_var': 0.0, 'cornish_fisher_var': 0.0, 'correction': 0.0,
                'skew': 0.0, 'excess_kurtosis': 0.0}

    mean = float(values.mean())
    skew = float(((values - mean) ** 3).mean() / std ** 3)
    kurtosis = float(((values - mean) ** 4).mean() / std ** 4)
    excess = kurtosis - 3.0

    z = float(norm.ppf(1.0 - confidence))
    gaussian = -(mean + z * std)

    # Cornish-Fisher expansion of the quantile.
    correction = (std / 6.0) * (z ** 2 - 1.0) * skew + \
                 (std ** 2 / 24.0) * (z ** 3 - 3.0 * z) * excess - \
                 (std ** 2 / 6.0) * (z ** 2 - 1.0) * skew ** 2
    adjusted = -(mean + z * std + correction)

    return {
        'gaussian_var': float(gaussian),
        'cornish_fisher_var': float(adjusted),
        'correction': float(adjusted - gaussian),
        'skew': skew,
        'excess_kurtosis': float(excess),
    }


def expected_shortfall(returns, confidence: float = 0.95) -> float:
    """
    Mean loss conditional on being in the worst ``1 - confidence`` tail.

    This is the number to size against. VaR says where the boundary sits;
    expected shortfall says how deep the losses go once you are past it, and
    for heavy tails it is the larger of the two.
    """
    values = _clean(returns, 'returns')
    _check_confidence(confidence)

    cutoff = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= cutoff]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


def drawdown_attribution(
    factors: Dict[str, Sequence],
    returns: Sequence,
) -> Dict[str, object]:
    """
    Identify the worst peak-to-trough period and rank factor contributions in it.

    Knowing the portfolio lost 30% is not actionable. Knowing that 70% of it
    happened during one window and was driven by a single factor is.

    :return: the peak, trough and recovery indices, the depth, and per-factor
        contributions accumulated over that window
    """
    names, matrix, y = _align(factors, returns)
    # The equity curve assumes per-period returns above -100%. Anything below
    # that is not a plausible return and would make the compounding meaningless,
    # so it is rejected rather than silently producing a spectacular fake drawdown.
    if np.any(y <= -1.0):
        raise ValueError(
            'returns must be greater than -100% per period to build an equity curve'
        )
    equity = np.cumprod(1.0 + y)

    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, equity / peaks - 1.0, 0.0)
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(equity[:trough + 1])) if trough > 0 else 0

    recovered = np.flatnonzero(equity[trough:] >= peaks[trough])
    recovery = int(trough + recovered[0]) if recovered.size else None

    window = y[peak:trough + 1]
    window_matrix = matrix[peak:trough + 1]

    contributions: Dict[str, float] = {}
    if window_matrix.shape[0] > window_matrix.shape[1] + 1:
        design = np.column_stack([np.ones(window_matrix.shape[0]), window_matrix])
        coefficients, *_ = np.linalg.lstsq(design, window, rcond=None)
        mean_factors = window_matrix.mean(axis=0)
        contributions = {
            name: float(coefficients[i + 1] * mean_factors[i]) for i, name in enumerate(names)
        }
    else:
        contributions = {name: 0.0 for name in names}

    return {
        'peak_index': peak,
        'trough_index': trough,
        'recovery_index': recovery,
        'peak_equity': float(equity[peak]),
        'trough_equity': float(equity[trough]),
        'max_drawdown': float(drawdowns[trough]),
        'bars_in_drawdown': int(trough - peak),
        'recovered': recovery is not None,
        'window_return': float(np.prod(1.0 + window) - 1.0),
        'contributions': contributions,
    }


def _clean(returns, label: str) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f'{label} is empty')
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(f'need at least 2 finite {label}, got {values.size}')
    return values


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f'confidence must be within (0, 1), got {confidence}')