"""
Aggregator for performance attribution.

Re-exports:
- TradeRecord / trade_pnl / execution_quality / attribute_trades: where the money came from
- univariate_attribution / ridge_attribution / attribution_report: which features earned it
"""

from .trades import (
    TradeRecord,
    attribute_trades,
    execution_quality,
    trade_pnl,
)
from .features import (
    attribution_report,
    ridge_attribution,
    univariate_attribution,
)

__all__ = [
    'TradeRecord',
    'trade_pnl',
    'execution_quality',
    'attribute_trades',
    'univariate_attribution',
    'ridge_attribution',
    'attribution_report',
]