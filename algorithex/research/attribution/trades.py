"""
Trade-level attribution.

Knowing a strategy made money is the least interesting thing about it. What
matters is *where* the money came from: was it one lucky trade, was it one
symbol, was it long or short, and did the exits actually capture the moves that
were available.

The last of those is the one most strategies lie to themselves about. Excursions
(maximum favourable and adverse movement while a trade was open) are measured
from the data you already have, and comparing the move you captured against the
move that was available is the difference between a strategy that exits well
and one that got lucky on a spike.

Inputs are plain mappings, not the database model, so this works equally well on
live results, a backtest store, or a list of dictionaries assembled by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    'TradeRecord',
    'trade_pnl',
    'execution_quality',
    'attribute_trades',
]

# Long trades gain when price rises, short trades when it falls.
_SIDE_SIGN = {'long': 1.0, 'buy': 1.0, 'short': -1.0, 'sell': -1.0}


@dataclass(frozen=True)
class TradeRecord:
    """One closed trade, reduced to the fields attribution needs."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    fees: float = 0.0
    opened_at: Optional[int] = None
    closed_at: Optional[int] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None

    @property
    def sign(self) -> float:
        return _SIDE_SIGN.get(str(self.side).lower(), 1.0)

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity * self.sign

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def duration(self) -> Optional[int]:
        if self.opened_at is None or self.closed_at is None:
            return None
        return int(self.closed_at) - int(self.opened_at)


def _as_record(item) -> TradeRecord:
    """Accept a TradeRecord, a mapping, or anything with the right attributes."""
    if isinstance(item, TradeRecord):
        return item

    def get(name: str, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    side = get('side', None) or get('type', 'long')
    # The database model exposes side indirectly via order direction.
    if side is None and get('buy_orders') is not None:
        side = 'long'

    quantity = get('quantity')
    if quantity is None:
        quantity = get('qty', 0.0)

    return TradeRecord(
        symbol=str(get('symbol', 'unknown')),
        side=str(side),
        entry_price=float(get('entry_price', 0.0) or 0.0),
        exit_price=float(get('exit_price', 0.0) or 0.0),
        quantity=float(quantity or 0.0),
        fees=float(get('fees', 0.0) or 0.0),
        opened_at=get('opened_at'),
        closed_at=get('closed_at'),
        mae=get('mae'),
        mfe=get('mfe'),
    )


def trade_pnl(trade) -> float:
    """
    Net profit or loss of a single trade.

    Fees are deducted because a strategy that is marginally profitable before
    costs is a strategy that loses money in practice.
    """
    return _as_record(trade).net_pnl


def execution_quality(trades: Sequence) -> Dict[str, object]:
    """
    How much of each trade's available move was actually captured.

    For a trade, the move available is its favourable excursion and the pain is
    its adverse one. ``capture_ratio`` divides captured profit by favourable
    excursion: near 1 means exits are well placed, near 0 means the trade gave
    back everything it had.

    Excursions are optional. When the caller does not supply them this returns
    ``available=False`` rather than inventing zeros, because a fabricated 0% is
    far more misleading than an honest "not measured".

    :param trades: trades carrying ``mae``/``mfe`` fields
    :return: aggregate capture statistics, or ``available=False``
    """
    records = [_as_record(t) for t in trades]
    measured = [r for r in records if r.mae is not None and r.mfe is not None]
    if not measured:
        return {'available': False, 'reason': 'no excursion data supplied'}

    ratios: List[float] = []
    give_back: List[float] = []
    for record in measured:
        favourable = abs(float(record.mfe) * record.quantity)
        adverse = abs(float(record.mae) * record.quantity)
        captured = record.gross_pnl
        if favourable > 0:
            ratios.append(captured / favourable)
        if favourable > 0:
            give_back.append((favourable - captured) / favourable)

    return {
        'available': True,
        'measured_trades': len(measured),
        'mean_capture_ratio': float(np.mean(ratios)) if ratios else 0.0,
        'median_capture_ratio': float(np.median(ratios)) if ratios else 0.0,
        'mean_give_back': float(np.mean(give_back)) if give_back else 0.0,
        'losing_share': float(
            np.mean([1.0 if r.gross_pnl < 0 else 0.0 for r in measured])
        ),
    }


def _group_totals(records: Sequence[TradeRecord], key) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        grouped.setdefault(key(record), []).append(record.net_pnl)

    summary: Dict[str, Dict[str, float]] = {}
    for name, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            'trades': int(array.size),
            'net_pnl': float(array.sum()),
            'mean_pnl': float(array.mean()),
            'wins': int(np.count_nonzero(array > 0)),
            'win_rate': float(np.count_nonzero(array > 0) / array.size) if array.size else 0.0,
        }
    return summary


def attribute_trades(trades: Iterable, time_bucket=None) -> Dict[str, object]:
    """
    Decompose a set of closed trades into per-symbol, per-side and per-period
    contributions.

    Concentration is reported explicitly, because a positive total can be one
    outlier trade in one symbol and nothing else. ``profit_concentration`` is
    the share of total profit carried by the single best trade; above roughly
    0.3 is the number worth arguing with.

    :param trades: trade records or mappings
    :param time_bucket: optional callable mapping a timestamp to a bucket
        label, e.g. ``lambda ts: ts // (86400 * 30)`` for month-long buckets
    :return: totals plus breakdowns by symbol, side and optional time bucket
    """
    records = [_as_record(t) for t in trades]
    if not records:
        raise ValueError('no trades to attribute')

    pnls = np.array([r.net_pnl for r in records], dtype=np.float64)
    gross = np.array([r.gross_pnl for r in records], dtype=np.float64)
    fees = np.array([r.fees for r in records], dtype=np.float64)

    total = float(pnls.sum())
    best = float(pnls.max())
    winning_sum = float(pnls[pnls > 0].sum())

    durations = [r.duration for r in records if r.duration is not None]
    report: Dict[str, object] = {
        'trades': len(records),
        'net_pnl': total,
        'gross_pnl': float(gross.sum()),
        'fees': float(fees.sum()),
        'fee_drag': float(fees.sum() / abs(gross.sum())) if gross.sum() else 0.0,
        'win_rate': float(np.count_nonzero(pnls > 0) / pnls.size),
        'average_pnl': float(pnls.mean()),
        'median_pnl': float(np.median(pnls)),
        'best_trade': best,
        'worst_trade': float(pnls.min()),
        # Share of gross profit contributed by the single best trade.
        'profit_concentration': float(best / winning_sum) if winning_sum > 0 else 0.0,
        'median_duration': float(np.median(durations)) if durations else None,
        'by_symbol': _group_totals(records, lambda r: r.symbol),
        'by_side': _group_totals(records, lambda r: str(r.side).lower()),
    }

    if time_bucket is not None:
        report['by_period'] = _group_totals(records, lambda r: str(time_bucket(r.closed_at)))

    report['execution'] = execution_quality(records)
    return report