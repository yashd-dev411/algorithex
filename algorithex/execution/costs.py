"""
Realistic trading costs.

A backtest that fills every order at the bar's close with no slippage is not a
backtest, it is a hypothesis about a market that does not exist. This module
supplies the missing physics: what a fill actually costs once you are competing
for liquidity with everyone else.

Four effects are modelled, roughly in the order they matter:

Commission
    The part the exchange bills you. Trivial to include and routinely omitted
    anyway.

Spread
    Crossing the bid-ask costs at least half the spread. Easy to forget that a
    *market* order pays it and a limit order pays it only when it fills.

Market impact
    The cost nobody models and everybody should. Your own order moves the price
    against you, and the damage grows with the square root of your size
    relative to the volume available — the standard Almgren square-root form.
    Linear impact badly over-penalises small orders and under-penalises large
    ones; the square root is why hedge funds use it.

Latency
    Deciding on bar *t* and filling on bar *t* close assumes you computed the
    signal before the bar finished forming. Usually you did not. Delaying the
    fill to the next bar is the cheapest honesty available.

Everything is expressed as a price adjustment in the direction that hurts you,
so a slippage model can never accidentally help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    'SIDE_SIGN',
    'FixedRateSlippage',
    'SpreadSlippage',
    'VolatilitySlippage',
    'SquareRootImpact',
    'CompositeSlippage',
    'FillCost',
    'cost_of_order',
    'funding_cost',
    'realized_volatility',
]

# Buying pushes the price up, selling pushes it down. Both cost the trader.
SIDE_SIGN = {'buy': 1.0, 'long': 1.0, 'sell': -1.0, 'short': -1.0}

_LONG_SIDE = ('buy', 'long')


def _sign(side) -> float:
    return SIDE_SIGN.get(str(side).lower(), 1.0)


class SlippageModel:
    """Base class: a slippage model reports a fractional price penalty."""

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedRateSlippage(SlippageModel):
    """A constant fractional penalty, e.g. 5 basis points per side."""

    bps: float = 5.0

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        if self.bps < 0:
            raise ValueError(f'bps must be non-negative, got {self.bps}')
        return self.bps / 10_000.0


@dataclass(frozen=True)
class SpreadSlippage(SlippageModel):
    """
    Half the bid-ask spread, the minimum a marketable order pays.

    :param bps: full spread in basis points; crossing it costs half
    :param marketable: when False the order rests and pays nothing unless the
        touch moves through it, which is optimistic but is what a passive
        limit order genuinely does
    """

    bps: float = 10.0
    marketable: bool = True

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        if self.bps < 0:
            raise ValueError(f'bps must be non-negative, got {self.bps}')
        if not self.marketable:
            return 0.0
        return self.bps / 2.0 / 10_000.0


@dataclass(frozen=True)
class VolatilitySlippage(SlippageModel):
    """
    Penalty proportional to recent volatility.

    Spread widens when the market moves, so a flat 10 bps spread assumption is
    optimistic precisely when it matters most. Use alongside, not instead of,
    a spread model.

    :param coefficient: penalty as a multiple of volatility
    :param floor: minimum volatility assumed, so a quiet stretch still costs
        something rather than becoming free
    """

    coefficient: float = 0.1
    floor: float = 1e-4

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        if self.coefficient < 0:
            raise ValueError(f'coefficient must be non-negative, got {self.coefficient}')
        return self.coefficient * max(float(volatility), self.floor)


@dataclass(frozen=True)
class SquareRootImpact(SlippageModel):
    """
    Almgren-style square-root market impact.

    The penalty grows with the square root of order size relative to the volume
    in the market, which is the empirical regularity behind most execution
    algos. Ten times the order costs roughly three times the penalty, not ten.

    :param coefficient: impact strength
    :param participation: fraction of bar volume assumed available to you
    """

    coefficient: float = 0.1
    participation: float = 0.1

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        if self.coefficient < 0:
            raise ValueError(f'coefficient must be non-negative, got {self.coefficient}')
        if not 0 < self.participation <= 1:
            raise ValueError(f'participation must be within (0, 1], got {self.participation}')

        size = abs(float(size))
        if size == 0:
            return 0.0

        # Without volume there is no denominator, so fall back to a fixed size
        # that corresponds to the configured participation of a nominal bar.
        reference = float(volume) if volume and volume > 0 else 1.0
        participation = min(size / (self.participation * reference), 1e6)
        return self.coefficient * np.sqrt(participation) * max(float(volatility), 0.0)


@dataclass(frozen=True)
class CompositeSlippage(SlippageModel):
    """Sum several slippage models, so their costs accumulate rather than mask."""

    models: tuple

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError('CompositeSlippage needs at least one model')

    def penalty(self, side, size: float, volatility: float = 0.0, volume: float = 0.0) -> float:
        return float(sum(
            model.penalty(side, size, volatility, volume) for model in self.models
        ))


@dataclass(frozen=True)
class FillCost:
    """Everything one fill cost, itemised rather than as a single number."""

    reference_price: float
    fill_price: float
    commission: float
    slippage_cost: float
    impact_cost: float
    side: str

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost + self.impact_cost

    @property
    def total_bps(self) -> float:
        notional = abs(self.reference_price * self.fill_price)
        if notional <= 0:
            return 0.0
        return self.total_cost / notional * 10_000.0


def cost_of_order(
    side,
    reference_price: float,
    size: float,
    slippage: Optional[SlippageModel] = None,
    impact: Optional[SlippageModel] = None,
    commission_rate: float = 0.0,
    commission_fixed: float = 0.0,
    volatility: float = 0.0,
    volume: float = 0.0,
) -> FillCost:
    """
    Price a fill against a reference, itemising every component.

    Returns the *adverse* price: a buy is pushed up, a sell is pushed down.
    A slippage model returning a negative penalty would make the fill better
    than the reference, so the result is floored at the reference.

    :param side: ``buy``/``long`` or ``sell``/``short``
    :param reference_price: price before costs, e.g. the bar close
    :param size: order quantity
    :param slippage: spread/volatility style model
    :param impact: market-impact model
    :param commission_rate: fractional fee, charged on notional
    :param commission_fixed: flat fee per order
    :param volatility: current realised volatility, for models that use it
    :param volume: bar volume, for impact models
    """
    if reference_price <= 0:
        raise ValueError(f'reference_price must be positive, got {reference_price}')
    if size < 0:
        raise ValueError(f'size must be non-negative, got {size}')

    direction = _sign(side)
    slippage_fraction = slippage.penalty(side, size, volatility, volume) if slippage else 0.0
    impact_fraction = impact.penalty(side, size, volatility, volume) if impact else 0.0

    # Penalties are always against the trader, never in their favour.
    slippage_fraction = max(float(slippage_fraction), 0.0)
    impact_fraction = max(float(impact_fraction), 0.0)

    fill_price = reference_price * (1.0 + direction * (slippage_fraction + impact_fraction))
    notional = reference_price * abs(size)
    commission = commission_rate * notional + commission_fixed

    move = abs(fill_price - reference_price) * abs(size)
    slippage_share = 0.0
    if move > 0 and (slippage_fraction + impact_fraction) > 0:
        slippage_share = slippage_fraction / (slippage_fraction + impact_fraction)

    return FillCost(
        reference_price=reference_price,
        fill_price=float(fill_price),
        commission=float(commission),
        slippage_cost=float(move * slippage_share),
        impact_cost=float(move * (1.0 - slippage_share)),
        side=str(side),
    )


def funding_cost(
    rate_per_interval: float,
    notional: float,
    intervals: float = 1.0,
    direction: str = 'long',
) -> float:
    """
    Cost of carrying a perpetual position across funding intervals.

    Longs pay when funding is positive, shorts receive it, and the sign flips
    when funding turns negative. Over a multi-month hold this quietly dominates
    everything else, which is why backtests that ignore it overstate returns for
    trend strategies specifically.

    :param rate_per_interval: funding rate, e.g. 0.0001 for 0.01%
    :param notional: position notional
    :param intervals: number of funding periods held
    :param direction: ``long`` pays positive funding, ``short`` receives it
    :return: positive is a cost, negative is a receipt
    """
    if intervals < 0:
        raise ValueError(f'intervals must be non-negative, got {intervals}')
    sign = 1.0 if str(direction).lower() in _LONG_SIDE else -1.0
    return float(rate_per_interval * notional * intervals * sign)


def realized_volatility(returns, window: int = 30) -> np.ndarray:
    """
    Trailing standard deviation of returns, for the models that consume it.

    Leading values are NaN rather than zero so a caller that forgets to skip
    warm-up gets no signal instead of falsely cheap fills.
    """
    series = np.asarray(returns, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f'returns must be 1D, got shape {series.shape}')
    if window < 2:
        raise ValueError(f'window must be at least 2, got {window}')

    out = np.full(series.size, np.nan)
    for i in range(window - 1, series.size):
        out[i] = float(np.std(series[i - window + 1:i + 1], ddof=1))
    return out