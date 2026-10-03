"""
Order book execution simulation.

Filling an order at the bar's close assumes you were the only participant. This
models the thing that actually determines your fill: the queue of orders ahead
of you at the touch, and how much depth sits behind it.

Two rules govern the simulation, and they are the two people forget:

Queue position
    A resting limit order does not fill when the price touches it. It fills
    when the market trades *through* it, and only orders ahead of it in the
    queue get filled. If you join a queue of 500 lots and someone sweeps 300,
    you are not one of the 300.

Depth, not price
    Size your order against the quantity available at each level. An order
    larger than the touch does not fill at the touch; it walks the ladder and
    pays more for the back of it.

The simulator returns the volume-weighted fill price and the fraction filled, so
a strategy can be scored on what it would realistically have achieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ['BookLevel', 'OrderBook', 'FillResult', 'simulate_fill']


@dataclass(frozen=True)
class BookLevel:
    """One price level: ``quantity`` lots available at ``price``."""

    price: float
    quantity: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f'price must be positive, got {self.price}')
        if self.quantity < 0:
            raise ValueError(f'quantity must be non-negative, got {self.quantity}')


@dataclass
class OrderBook:
    """
    A two-sided book.

    Levels are held best-price-first; the constructor does not sort them, so
    passing an unsorted ladder is a caller bug that shows up as a nonsense fill
    rather than being quietly fixed.

    :param bids: buy-side levels, best (highest) price first
    :param asks: sell-side levels, best (lowest) price first
    """

    bids: List[BookLevel] = field(default_factory=list)
    asks: List[BookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def is_crossed(self) -> bool:
        """True when the bid is at or above the ask, which no real book is."""
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid >= self.best_ask
        )


@dataclass(frozen=True)
class FillResult:
    """What a simulated order actually achieved."""

    requested: float
    filled: float
    volume_weighted_price: Optional[float]
    levels_consumed: int

    @property
    def fill_ratio(self) -> float:
        return self.filled / self.requested if self.requested > 0 else 0.0

    @property
    def fully_filled(self) -> bool:
        return self.filled >= self.requested - 1e-12


def simulate_fill(
    book: OrderBook,
    side: str,
    size: float,
    queue_ahead: float = 0.0,
    marketable: bool = True,
) -> FillResult:
    """
    Fill an order against a book, walking the ladder.

    A buy consumes asks, a sell consumes bids — always taking the worse price
    available, which is what a taker pays.

    :param book: the ladder to fill against
    :param side: ``buy``/``long`` or ``sell``/``short``
    :param size: requested quantity
    :param queue_ahead: quantity resting ahead of a *passive* order at the
        touch. Only a marketable order ignores it.
    :param marketable: cross the spread now. A passive order only fills from
        the quantity behind its own queue position, so a large order joining a
        large queue may not fill at all.
    :return: the fill, including a volume-weighted price and how much of the
        order actually executed
    """
    if size < 0:
        raise ValueError(f'size must be non-negative, got {size}')
    if queue_ahead < 0:
        raise ValueError(f'queue_ahead must be non-negative, got {queue_ahead}')

    is_buy = str(side).lower() in ('buy', 'long')
    levels = list(book.asks if is_buy else book.bids)

    if size == 0:
        return FillResult(requested=0.0, filled=0.0, volume_weighted_price=None, levels_consumed=0)

    if not levels:
        return FillResult(
            requested=float(size), filled=0.0,
            volume_weighted_price=None, levels_consumed=0,
        )

    remaining = float(size)

    if not marketable:
        # A passive order sits behind everything resting at the touch.
        available_at_touch = max(levels[0].quantity - queue_ahead, 0.0)
        filled = min(remaining, available_at_touch)
        price = levels[0].price if filled > 0 else None
        return FillResult(
            requested=float(size), filled=float(filled),
            volume_weighted_price=price, levels_consumed=1 if filled > 0 else 0,
        )

    notional = 0.0
    filled = 0.0
    consumed = 0
    for level in levels:
        if remaining <= 0:
            break
        take = min(remaining, level.quantity)
        notional += take * level.price
        filled += take
        remaining -= take
        consumed += 1

    vwap = notional / filled if filled > 0 else None
    return FillResult(
        requested=float(size), filled=float(filled),
        volume_weighted_price=None if vwap is None else float(vwap),
        levels_consumed=consumed,
    )