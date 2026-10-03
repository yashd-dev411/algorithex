"""
Aggregator for walk-forward validation.

Re-exports:
- rolling_windows / anchored_windows: fixed-length and expanding train/test windows
- purge_train_indices: remove training observations whose label reaches into test
- purged_walk_forward: anchored or rolling windows with leakage removed
- purged_kfold / combinatorial_purged_split: purged K-fold and combinatorial paths
- stitch_oos_returns / performance_metrics / evaluate_windows: scoring
"""

from .splits import (
    Window,
    anchored_windows,
    combinatorial_paths,
    combinatorial_purged_split,
    purge_train_indices,
    purged_kfold,
    purged_walk_forward,
    rolling_windows,
)
from .evaluate import (
    PathResult,
    evaluate_windows,
    performance_metrics,
    stitch_oos_returns,
)

__all__ = [
    'Window',
    'rolling_windows',
    'anchored_windows',
    'purge_train_indices',
    'purged_walk_forward',
    'purged_kfold',
    'combinatorial_paths',
    'combinatorial_purged_split',
    'stitch_oos_returns',
    'performance_metrics',
    'PathResult',
    'evaluate_windows',
]