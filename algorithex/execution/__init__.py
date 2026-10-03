"""
Aggregator for execution realism.

Re-exports:
- FixedRateSlippage / SpreadSlippage / VolatilitySlippage / SquareRootImpact /
  CompositeSlippage: what a fill actually costs
- cost_of_order / funding_cost: itemised cost of one order
- OrderBook / BookLevel / simulate_fill: queue-aware fills against real depth
- assess_parity / fill_deviation / compare_returns: backtest-vs-live drift
- twap_schedule / pov_schedule / vwap_schedule / is_optimal / adaptive_pov_schedule:
  how to work an order rather than firing it whole
- simulate_execution / compare_algorithms: score a schedule and pick one
"""

from .costs import (
    SIDE_SIGN,
    CompositeSlippage,
    FillCost,
    FixedRateSlippage,
    SlippageModel,
    SpreadSlippage,
    SquareRootImpact,
    VolatilitySlippage,
    cost_of_order,
    funding_cost,
    realized_volatility,
)
from .book import BookLevel, FillResult, OrderBook, simulate_fill
from .parity import ParityReport, assess_parity, compare_returns, fill_deviation
from .algos import (
    AlgorithmComparison,
    ChildOrder,
    ExecutionParams,
    ExecutionPlan,
    ExecutionResult,
    adaptive_pov_schedule,
    benchmark_curve,
    compare_algorithms,
    is_optimal,
    pov_schedule,
    schedule_by_name,
    simulate_execution,
    twap_schedule,
    vwap_schedule,
)

__all__ = [
    'SIDE_SIGN',
    'SlippageModel',
    'FixedRateSlippage',
    'SpreadSlippage',
    'VolatilitySlippage',
    'SquareRootImpact',
    'CompositeSlippage',
    'FillCost',
    'cost_of_order',
    'funding_cost',
    'realized_volatility',
    'BookLevel',
    'OrderBook',
    'FillResult',
    'simulate_fill',
    'ParityReport',
    'fill_deviation',
    'assess_parity',
    'compare_returns',
    'ExecutionParams',
    'ChildOrder',
    'ExecutionPlan',
    'ExecutionResult',
    'AlgorithmComparison',
    'twap_schedule',
    'pov_schedule',
    'vwap_schedule',
    'is_optimal',
    'adaptive_pov_schedule',
    'benchmark_curve',
    'schedule_by_name',
    'simulate_execution',
    'compare_algorithms',
]