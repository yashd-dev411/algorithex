"""
Backtest-versus-live parity.

The question this answers is the one nobody asks until it is too late: *is my
backtest lying to me, and by how much?*

It compares what a backtest assumed about each fill against what actually
happened, and reports the deviation. The statistic that matters is the **mean
signed slippage**: if realised fills are consistently worse than assumed, the
backtest is optimistic by that much on every trade, and that gap compounds
across a strategy's whole life.

A symmetric deviation around zero is noise. A consistently one-sided deviation
is a modelling error, and :func:`assess_parity` is built to detect exactly that
distinction rather than averaging it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ['ParityReport', 'fill_deviation', 'assess_parity', 'compare_returns']


@dataclass
class ParityReport:
    """How far actual execution drifted from what the backtest assumed."""

    observations: int
    mean_signed_bps: float
    median_signed_bps: float
    mean_absolute_bps: float
    worst_bps: float
    bias_ratio: float
    systematic: bool
    verdict: str
    per_symbol: Dict[str, Dict[str, float]]


def fill_deviation(
    assumed_prices: Sequence[float],
    realized_prices: Sequence[float],
    sides: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Signed slippage per trade, in basis points.

    Positive means reality was **worse** than the backtest assumed, which is the
    direction that matters. Side matters because a sell that fills higher than
    expected is a loss, while a buy filling higher is a gain — measuring both
    with the same sign would cancel a systematic error into invisibility.

    :param assumed_prices: price the backtest used
    :param realized_prices: price actually achieved
    :param sides: ``buy``/``sell`` per trade; assumed all buys if omitted
    :return: signed basis points, positive meaning worse than assumed
    """
    assumed = np.asarray(assumed_prices, dtype=np.float64)
    realized = np.asarray(realized_prices, dtype=np.float64)
    if assumed.shape != realized.shape:
        raise ValueError(
            f'assumed ({assumed.size}) and realized ({realized.size}) differ in length'
        )
    if assumed.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(assumed == 0):
        raise ValueError('assumed prices must be non-zero to measure a percentage')

    if sides is None:
        signs = np.ones(assumed.size, dtype=np.float64)
    else:
        side_list = [str(s).lower() for s in sides]
        if len(side_list) != assumed.size:
            raise ValueError('sides must match the number of fills')
        signs = np.array(
            [-1.0 if s in ('sell', 'short') else 1.0 for s in side_list],
            dtype=np.float64,
        )

    relative = (realized - assumed) / assumed
    return signs * relative * 10_000.0


def assess_parity(
    assumed_prices: Sequence[float],
    realized_prices: Sequence[float],
    sides: Optional[Sequence[str]] = None,
    symbols: Optional[Sequence[str]] = None,
    bias_tolerance_bps: float = 2.0,
) -> ParityReport:
    """
    Measure and judge how far live execution drifted from the backtest.

    :param assumed_prices: price the backtest used
    :param realized_prices: price actually achieved
    :param sides: side per trade
    :param symbols: symbol per trade, for a per-symbol breakdown
    :param bias_tolerance_bps: mean deviation beyond which the backtest is
        called systematically optimistic
    """
    deviation = fill_deviation(assumed_prices, realized_prices, sides)

    if deviation.size == 0:
        return ParityReport(
            observations=0, mean_signed_bps=0.0, median_signed_bps=0.0,
            mean_absolute_bps=0.0, worst_bps=0.0, bias_ratio=0.0,
            systematic=False, verdict='no fills to compare',
            per_symbol={},
        )

    mean_signed = float(deviation.mean())
    mean_absolute = float(np.abs(deviation).mean())

    # Share of the total absolute deviation carried by the net direction. Near
    # 1.0 means every miss went the same way, which is an error, not noise.
    bias_ratio = float(abs(mean_signed) / mean_absolute) if mean_absolute > 0 else 0.0
    systematic = bool(mean_signed > bias_tolerance_bps and bias_ratio > 0.3)

    if systematic:
        verdict = (
            f'backtest is optimistic by {mean_signed:.1f} bps per fill '
            f'({bias_ratio * 100:.0f}% of the total deviation is one-sided)'
        )
    elif abs(mean_signed) > bias_tolerance_bps:
        verdict = (
            f'live fills are {abs(mean_signed):.1f} bps *better* than assumed, '
            'which is also worth explaining'
        )
    else:
        verdict = f'live fills are within {bias_tolerance_bps:.1f} bps of the backtest'

    per_symbol: Dict[str, Dict[str, float]] = {}
    if symbols is not None:
        names = list(symbols)
        if len(names) != deviation.size:
            raise ValueError('symbols must match the number of fills')
        for name in dict.fromkeys(names):
            mask = np.array([n == name for n in names])
            values = deviation[mask]
            per_symbol[name] = {
                'observations': int(values.size),
                'mean_signed_bps': float(values.mean()),
                'mean_absolute_bps': float(np.abs(values).mean()),
            }

    return ParityReport(
        observations=int(deviation.size),
        mean_signed_bps=mean_signed,
        median_signed_bps=float(np.median(deviation)),
        mean_absolute_bps=mean_absolute,
        worst_bps=float(deviation.max()),
        bias_ratio=bias_ratio,
        systematic=systematic,
        verdict=verdict,
        per_symbol=per_symbol,
    )


def compare_returns(
    backtest_returns,
    live_returns,
    periods_per_year: int = 365,
) -> Dict[str, object]:
    """
    Compare a backtest's return series against what actually happened.

    A backtest can match on headline return and still be wrong: the live series
    may simply be more volatile at the same mean. Reporting the gap in Sharpe
    alongside the gap in return catches that.

    :param backtest_returns: per-bar returns the backtest produced
    :param live_returns: per-bar returns actually realised
    """
    from .costs import realized_volatility  # local import keeps the module light

    expected = np.asarray(backtest_returns, dtype=np.float64)
    actual = np.asarray(live_returns, dtype=np.float64)
    if expected.shape != actual.shape:
        raise ValueError(
            f'backtest ({expected.size}) and live ({actual.size}) series differ in length'
        )
    if expected.size < 2:
        raise ValueError('need at least two observations to compare')

    scale = np.sqrt(periods_per_year)

    def summarise(values: np.ndarray) -> Dict[str, float]:
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        return {
            'total_return': float(np.prod(1.0 + values) - 1.0),
            'sharpe': float(mean / std * scale) if std > 0 else float('nan'),
            'volatility': float(std * scale),
        }

    backtest = summarise(expected)
    live = summarise(actual)

    correlation = (
        float(np.corrcoef(expected, actual)[0, 1]) if expected.std() > 0 and actual.std() > 0
        else float('nan')
    )

    return {
        'backtest': backtest,
        'live': live,
        'return_gap': backtest['total_return'] - live['total_return'],
        'sharpe_gap': backtest['sharpe'] - live['sharpe'],
        'volatility_ratio': (
            live['volatility'] / backtest['volatility']
            if backtest['volatility'] > 0 else float('nan')
        ),
        'correlation': correlation,
        'aligned': bool(correlation > 0.8),
        'realized_volatility': realized_volatility(actual),
    }