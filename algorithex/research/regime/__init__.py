"""
Aggregator for regime detection.

Re-exports:
- GaussianHMM: continuous-observation hidden Markov model (Baum-Welch + Viterbi)
- detect_change_points / segments: where the environment changed
- realized_volatility / regime_volatility / volatility_target_size /
  drawdown_throttle / regime_position_size: regime-aware sizing
"""

from .markov import GaussianHMM
from .changepoint import (
    ChangePoint,
    best_split,
    detect_change_points,
    segment_cost,
    segments,
)
from .adaptive import (
    drawdown_throttle,
    realized_volatility,
    regime_position_size,
    regime_volatility,
    volatility_target_size,
)

__all__ = [
    'GaussianHMM',
    'ChangePoint',
    'segment_cost',
    'best_split',
    'detect_change_points',
    'segments',
    'realized_volatility',
    'regime_volatility',
    'volatility_target_size',
    'drawdown_throttle',
    'regime_position_size',
]