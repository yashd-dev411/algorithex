"""
Regime-aware position sizing.

Fixed position sizes are the quiet assumption behind most backtest failures: a
strategy sized for calm conditions is oversized exactly when it starts working,
and undersized when conditions are hostile. Sizing to a *measured* target
volatility removes that coupling, and conditioning on the detected regime lets
the target itself move.

Everything here is a multiplier in ``[lower, upper]``. Keeping sizing as a
bounded multiplier rather than an absolute position keeps it composable with
whatever the strategy or exchange already does, and means a sizing bug
degrades to a smaller trade rather than a blown account.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    'realized_volatility',
    'regime_volatility',
    'volatility_target_size',
    'drawdown_throttle',
    'regime_position_size',
]


def realized_volatility(returns, window: int = 30, min_periods: Optional[int] = None):
    """
    Trailing standard deviation of per-bar returns.

    :param returns: per-bar returns
    :param window: lookback in bars
    :param min_periods: minimum observations before a value is emitted;
        defaults to ``window``. The leading values are NaN rather than zero,
        so a caller that ignores them gets no signal instead of false calm.
    """
    series = np.asarray(returns, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f'returns must be 1D, got shape {series.shape}')
    if window < 2:
        raise ValueError(f'window must be at least 2, got {window}')

    minimum = window if min_periods is None else min_periods
    if minimum < 2:
        raise ValueError(f'min_periods must be at least 2, got {minimum}')

    out = np.full(series.size, np.nan)
    for i in range(minimum - 1, series.size):
        out[i] = float(np.std(series[i - minimum + 1:i + 1], ddof=1))
    return out


def regime_volatility(states, returns, min_periods: int = 5) -> dict:
    """
    Mean realised volatility per detected regime.

    Volatility clusters, so this is usually strongly ordered across states:
    that ordering is the signal. A regime whose volatility is not meaningfully
    different from the others is not worth conditioning on, which is why the
    spread is reported alongside the levels.

    :param states: regime label per bar
    :param returns: per-bar returns
    :param min_periods: observations required before a regime is scored
    :return: mapping of state label to statistics, plus a ``_spread`` entry
    """
    labels = np.asarray(states)
    series = np.asarray(returns, dtype=np.float64)
    if labels.shape[0] != series.shape[0]:
        raise ValueError(
            f'states ({labels.shape[0]}) and returns ({series.shape[0]}) '
            'must be the same length'
        )

    summary: dict = {}
    for state in np.unique(labels):
        mask = labels == state
        values = series[mask]
        if values.size < min_periods:
            continue
        summary[state] = {
            'observations': int(values.size),
            'volatility': float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            'mean_return': float(np.mean(values)),
        }

    levels = [v['volatility'] for v in summary.values() if v['volatility'] > 0]
    summary['_spread'] = {
        'min': float(min(levels)) if levels else 0.0,
        'max': float(max(levels)) if levels else 0.0,
        'ratio': float(max(levels) / min(levels)) if len(levels) > 1 else 1.0,
    }
    return summary


def volatility_target_size(
    realized_vol,
    target_vol: float = 0.01,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """
    Scale a position so its expected volatility sits near ``target_vol``.

    The multiplier is ``target_vol / observed_vol``: double the observed
    volatility and you halve the position. Clipped to ``[lower, upper]``, with
    NaN inputs passing through as NaN so warm-up periods stay visibly warm-up.

    :param realized_vol: trailing volatility per bar
    :param target_vol: desired per-bar volatility
    :param lower: floor on the multiplier, e.g. 0.0 to allow full de-risking
    :param upper: cap, which must not exceed leverage you actually have
    """
    observed = np.asarray(realized_vol, dtype=np.float64)
    if target_vol <= 0:
        raise ValueError(f'target_vol must be positive, got {target_vol}')
    if lower > upper:
        raise ValueError(f'lower ({lower}) must not exceed upper ({upper})')

    out = np.full(observed.shape, np.nan)
    usable = np.isfinite(observed) & (observed > 0)
    out[usable] = np.clip(target_vol / observed[usable], lower, upper)
    return out


def drawdown_throttle(
    equity_or_returns,
    max_drawdown: float = 0.2,
    window: Optional[int] = None,
    lower: float = 0.0,
    recovery: bool = True,
) -> np.ndarray:
    """
    Reduce size as equity falls away from its peak.

    The peak is trailing rather than all-time by default, so a strategy that
    recovers can earn its size back instead of staying permanently throttled by
    a high it set months ago. Pass ``window=None`` for a strict all-time peak.

    :param equity_or_returns: an equity curve, or per-bar returns to compound
    :param max_drawdown: throttle begins once drawdown exceeds this fraction
    :param window: bars of trailing peak; defaults to the whole series
    :param lower: floor on the multiplier
    :param recovery: blend size back toward full as drawdown heals
    """
    if max_drawdown <= 0:
        raise ValueError(f'max_drawdown must be positive, got {max_drawdown}')
    if window is not None and window < 1:
        raise ValueError(f'window must be at least 1, got {window}')
    if lower > 1.0:
        raise ValueError(f'lower must not exceed 1.0, got {lower}')

    series = np.asarray(equity_or_returns, dtype=np.float64)
    if series.ndim != 1 or series.size == 0:
        raise ValueError('need a non-empty 1D equity or return series')
    if not np.all(np.isfinite(series)):
        raise ValueError('series contains NaN or infinite values')

    if np.any(series <= 0) and series.max() <= 1.0:
        # Treat as returns and compound them into an equity curve.
        series = np.cumprod(1.0 + series)

    if window is None or window >= series.size:
        peak = np.maximum.accumulate(series)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(series, window)
        # Front-pad with the first value so the rolling maximum stays aligned
        # one-to-one with the input instead of starting `window - 1` bars late.
        padding = np.full((window - 1, window), series[0])
        peak = np.maximum.reduce(np.concatenate([padding, windows], axis=0), axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak > 0, 1.0 - series / peak, 0.0)
    drawdown = np.clip(drawdown, 0.0, 1.0)

    scale = np.ones_like(drawdown)
    breached = drawdown > max_drawdown
    if recovery:
        healing = np.clip((max_drawdown - drawdown) / max_drawdown, 0.0, 1.0)
        scale = np.where(breached, lower + (1.0 - lower) * healing, 1.0)
    else:
        scale = np.where(breached, lower, 1.0)
    return scale


def regime_position_size(
    states,
    returns,
    target_vol: float = 0.01,
    window: int = 30,
    lower: float = 0.0,
    upper: float = 1.0,
    warmup_state: float = 0.0,
) -> np.ndarray:
    """
    Size each bar by the volatility of the regime it falls in.

    Realised volatility is measured *within* each regime rather than across
    the whole series, so the model sizes a bar using the conditions of its own
    regime instead of a trailing average that still contains the calm period it
    just left. Volatility clustering makes that distinction material.

    :param states: regime label per bar
    :param returns: per-bar returns
    :param target_vol: desired per-bar volatility
    :param window: lookback used within each regime
    :param lower: floor on the multiplier
    :param upper: cap on the multiplier
    :param warmup_state: size applied before any regime has enough history
    """
    labels = np.asarray(states)
    series = np.asarray(returns, dtype=np.float64)
    if labels.shape[0] != series.shape[0]:
        raise ValueError('states and returns must be the same length')
    if target_vol <= 0:
        raise ValueError(f'target_vol must be positive, got {target_vol}')

    size = np.full(series.shape, warmup_state, dtype=np.float64)

    for state in np.unique(labels):
        positions = np.flatnonzero(labels == state)
        if positions.size < window:
            continue
        # rolling volatility within this regime's own observations
        values = series[positions]
        rolling = np.full(values.size, np.nan)
        for i in range(window - 1, values.size):
            rolling[i] = float(np.std(values[i - window + 1:i + 1], ddof=1))
        size[positions] = volatility_target_size(
            rolling, target_vol=target_vol, lower=lower, upper=upper
        )

    # Bars with insufficient history inside their regime fall back to warmup.
    unresolved = ~np.isfinite(size)
    size[unresolved] = warmup_state
    return size