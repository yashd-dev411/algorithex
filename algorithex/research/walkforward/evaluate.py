"""
Scoring walk-forward and combinatorial validation results.

The hard part of out-of-sample evaluation is not the metrics, it is the
stitching. Anchored test windows overlap in time, and combinatorial paths reuse
bars across paths. Naively concatenating every test window double counts those
bars, and reporting the best path afterwards is precisely the selection bias
this package exists to prevent. :func:`stitch_oos_returns` collapses everything
onto one chronological series where each bar is counted at most once, and
:func:`evaluate_windows` reports the distribution across paths rather than the
maximum.

All functions take a per-bar return series aligned to the same index space the
splits were built from, so they compose with any strategy without knowing
anything about the trading engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .splits import Window

__all__ = [
    'stitch_oos_returns',
    'performance_metrics',
    'PathResult',
    'evaluate_windows',
]


def _as_returns(returns) -> np.ndarray:
    array = np.asarray(returns, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f'returns must be one dimensional, got shape {array.shape}')
    return array


def stitch_oos_returns(windows: Sequence[Window], returns) -> np.ndarray:
    """
    Collapse every test window into one chronological, duplicate-free series.

    A bar appearing in three test windows is still one observation, so it is
    averaged across the paths that saw it. Averaging rather than taking the
    first occurrence keeps the result stable when paths disagree about a bar.

    :param windows: the validation windows
    :param returns: per-bar returns aligned to the window index space
    :return: the out-of-sample returns, in chronological order
    """
    returns = _as_returns(returns)

    totals: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for window in windows:
        for position in window.test:
            index = int(position)
            if index < 0 or index >= returns.size:
                raise IndexError(
                    f'window references bar {index} but returns has '
                    f'{returns.size} entries'
                )
            totals[index] = totals.get(index, 0.0) + float(returns[index])
            counts[index] = counts.get(index, 0) + 1

    if not totals:
        return np.empty(0, dtype=np.float64)

    ordered = sorted(totals)
    stitched = np.array([totals[i] / counts[i] for i in ordered], dtype=np.float64)
    return stitched


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(running_peak > 0, equity / running_peak - 1.0, 0.0)
    return float(drawdown.min())


def performance_metrics(returns, periods_per_year: int = 365) -> Dict[str, float]:
    """
    Risk-adjusted statistics for a per-bar return series.

    Every statistic degrades gracefully on an empty or constant series rather
    than dividing by zero, because a validation run over a short window really
    does produce those and crashing on them hides the result.

    :param returns: per-bar returns (simple, not logarithmic)
    :param periods_per_year: bars per year, used to annualise; 365 suits
        crypto, 252 suits equities
    """
    returns = _as_returns(returns)
    n = returns.size
    nan = float('nan')

    if n == 0:
        return {
            'observations': 0, 'total_return': 0.0, 'mean': nan,
            'volatility': nan, 'sharpe': nan, 'sortino': nan, 'calmar': nan,
            'max_drawdown': 0.0, 'hit_rate': nan, 'best': nan, 'worst': nan,
            'annualised_return': nan, 'annualised_volatility': nan,
        }

    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    scale = float(np.sqrt(periods_per_year))

    if std > 0:
        sharpe = mean / std * scale
    else:
        sharpe = nan

    downside = returns[returns < 0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    sortino = mean / downside_dev * scale if downside_dev > 0 else nan

    equity = np.cumprod(1.0 + returns)
    max_drawdown = _max_drawdown(equity)

    years = n / periods_per_year
    if years > 0 and equity[-1] > 0:
        annualised_return = float(equity[-1] ** (1.0 / years) - 1.0)
    else:
        annualised_return = nan

    if max_drawdown < 0:
        calmar = annualised_return / abs(max_drawdown)
    else:
        calmar = nan

    return {
        'observations': n,
        'total_return': float(equity[-1] - 1.0),
        'annualised_return': annualised_return,
        'mean': mean,
        'volatility': std * scale,
        'annualised_volatility': std * scale,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'hit_rate': float(np.count_nonzero(returns > 0) / n),
        'best': float(returns.max()),
        'worst': float(returns.min()),
    }


@dataclass
class PathResult:
    """Metrics for a single validation path."""

    index: int
    test_size: int
    metrics: Dict[str, float] = field(default_factory=dict)


def _finite(values: List[float]) -> List[float]:
    return [v for v in values if np.isfinite(v)]


def evaluate_windows(
    windows: Sequence[Window],
    returns,
    periods_per_year: int = 365,
) -> Dict[str, object]:
    """
    Score every validation path and summarise the distribution.

    The headline numbers come from the stitched series, which is the honest
    one. The per-path spread matters just as much: a strategy whose Sharpe is
    2.0 on the stitched series but negative on most individual paths has found
    a partition, not an edge.

    :return: dict with ``stitched`` metrics, per-path results and the
        distribution of each headline statistic across paths
    """
    returns = _as_returns(returns)
    windows = list(windows)
    if not windows:
        raise ValueError('no windows to evaluate')

    paths: List[PathResult] = []
    for window in windows:
        metrics = performance_metrics(returns[window.test], periods_per_year)
        paths.append(
            PathResult(
                index=window.index,
                test_size=int(window.test.size),
                metrics=metrics,
            )
        )

    stitched = stitch_oos_returns(windows, returns)
    overall = performance_metrics(stitched, periods_per_year)

    spread: Dict[str, Dict[str, float]] = {}
    for key in ('sharpe', 'sortino', 'total_return', 'max_drawdown', 'hit_rate'):
        values = _finite([p.metrics.get(key, float('nan')) for p in paths])
        if not values:
            spread[key] = {'min': float('nan'), 'median': float('nan'), 'max': float('nan')}
            continue
        spread[key] = {
            'min': float(np.min(values)),
            'median': float(np.median(values)),
            'max': float(np.max(values)),
        }

    positive = sum(1 for p in paths if np.isfinite(p.metrics.get('sharpe', np.nan)) and p.metrics['sharpe'] > 0)

    return {
        'stitched': overall,
        'paths': paths,
        'path_count': len(paths),
        'profitable_path_ratio': positive / len(paths) if paths else float('nan'),
        'spread': spread,
        'oos_returns': stitched,
    }