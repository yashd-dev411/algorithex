"""
Train/test splits for time-series validation.

Everything here is positional index arithmetic. No trading engine, database or
network state is touched, so the same splits can drive a backtest, a live
walk-forward run, or a plain NumPy array. That separation is deliberate: a
split is a statement about *which bars a model may learn from*, and it should
be checkable without running a single trade.

The vocabulary:

train / test
    Two disjoint index arrays. The model sees ``train``; only ``test``
    measures it.

anchored vs rolling
    An anchored window always starts training at observation zero and grows,
    so every test set is scored against all history seen so far. A rolling
    window keeps training a fixed length and discards old data. Anchored is
    the better default for non-stationary series; rolling is the better default
    when regimes change so much that ancient data is misleading.

embargo (a.k.a. purge)
    A bar's outcome is not known at that bar. If a training bar sits at the end
    of the training set, its label may reach forward into the test set and
    leak. Purging drops training observations whose label window overlaps the
    test set, and the embargo additionally drops a band immediately *after* the
    test set, which is what serial-correlation leakage looks like in practice.

combinatorial purged cross-validation
    Instead of one train/test partition, split the series into ``n_splits``
    contiguous groups, choose ``n_test_splits`` of them to form each test set,
    and train on the purged remainder. This yields many overlapping test
    paths through the data, so a strategy is judged on how it performs along
    many routes rather than one arbitrary one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Sequence

import numpy as np

__all__ = [
    'Window',
    'rolling_windows',
    'anchored_windows',
    'purge_train_indices',
    'purged_walk_forward',
    'purged_kfold',
    'combinatorial_paths',
    'combinatorial_purged_split',
]


@dataclass(frozen=True)
class Window:
    """One train/test partition, as positional indices into a series."""

    index: int
    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'train', np.asarray(self.train, dtype=np.int64))
        object.__setattr__(self, 'test', np.asarray(self.test, dtype=np.int64))

    @property
    def train_size(self) -> int:
        return int(self.train.size)

    @property
    def test_size(self) -> int:
        return int(self.test.size)

    @property
    def test_bounds(self) -> tuple:
        """Inclusive (start, end) of the test window."""
        return int(self.test.min()), int(self.test.max())

    def __repr__(self) -> str:
        train_lo, train_hi = int(self.train.min()), int(self.train.max())
        test_lo, test_hi = int(self.test.min()), int(self.test.max())
        return (
            f'Window(index={self.index}, '
            f'train=[{train_lo}..{train_hi}], test=[{test_lo}..{test_hi}])'
        )


def _validate(n_obs: int, train_size: int, test_size: int) -> None:
    if n_obs <= 0:
        raise ValueError(f'n_obs must be positive, got {n_obs}')
    if train_size <= 0:
        raise ValueError(f'train_size must be positive, got {train_size}')
    if test_size <= 0:
        raise ValueError(f'test_size must be positive, got {test_size}')


def _windows(
    n_obs: int,
    train_size: int,
    test_size: int,
    step: int,
    min_train: int,
    anchored: bool,
) -> Iterator[Window]:
    _validate(n_obs, train_size, test_size)
    if step <= 0:
        raise ValueError(f'step must be positive, got {step}')

    index = 0
    boundary = min_train
    while boundary + test_size <= n_obs:
        train_lo = 0 if anchored else max(0, boundary - train_size)
        window = Window(
            index=index,
            train=np.arange(train_lo, boundary),
            test=np.arange(boundary, boundary + test_size),
        )
        yield window
        index += 1
        boundary += step


def rolling_windows(
    n_obs: int,
    train_size: int,
    test_size: int,
    step: int = None,
    min_train: int = None,
) -> List[Window]:
    """
    Fixed-length training windows that slide forward through the series.

    :param n_obs: total number of observations available
    :param train_size: length of each training window
    :param test_size: length of each test window
    :param step: bars between consecutive windows; defaults to ``test_size``,
        which makes test sets disjoint and therefore straightforward to stitch
    :param min_train: smallest training window, for growing from less than
        ``train_size`` bars of history; defaults to ``train_size``
    """
    return list(
        _windows(
            n_obs,
            train_size,
            test_size,
            test_size if step is None else step,
            train_size if min_train is None else min_train,
            anchored=False,
        )
    )


def anchored_windows(
    n_obs: int,
    train_size: int,
    test_size: int,
    step: int = None,
    min_train: int = None,
) -> List[Window]:
    """
    Expanding training windows: every fit uses all history up to the boundary.

    ``train_size`` acts as the starting history rather than a cap. This is the
    right shape for markets, where old data stops being representative but
    never stops being evidence.
    """
    return list(
        _windows(
            n_obs,
            train_size,
            test_size,
            test_size if step is None else step,
            train_size if min_train is None else min_train,
            anchored=True,
        )
    )


def purge_train_indices(
    train: Sequence[int],
    test: Sequence[int],
    embargo: int = 0,
) -> np.ndarray:
    """
    Drop training observations whose label window can reach into the test set.

    An observation at bar ``i`` is assumed to have a label extending forward to
    ``i + embargo``. It survives only when that whole window clears the test
    block on both sides.

    :param train: candidate training indices
    :param test: indices reserved for testing
    :param embargo: label length in bars; 0 keeps every training observation
    :return: the surviving training indices, still sorted
    """
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if test.size == 0 or embargo <= 0:
        return train
    if train.size == 0:
        return train

    test_lo, test_hi = int(test.min()), int(test.max())
    clears_before = (train + embargo) < test_lo
    clears_after = train > test_hi
    return train[clears_before | clears_after]


def purged_walk_forward(
    n_obs: int,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    step: int = None,
    min_train: int = None,
    anchored: bool = True,
) -> List[Window]:
    """
    Walk-forward windows with leakage removed from each training set.

    The embargo is applied around every test block, so no training label
    overlaps the test period it is being scored against.

    :param embargo: label length in bars; the single most important parameter
        here, since setting it to 0 reintroduces lookahead bias
    :param anchored: expanding (default) or fixed-length training windows
    """
    make = anchored_windows if anchored else rolling_windows
    raw = make(n_obs, train_size, test_size, step=step, min_train=min_train)

    windows: List[Window] = []
    for window in raw:
        windows.append(
            Window(
                index=window.index,
                train=purge_train_indices(window.train, window.test, embargo),
                test=window.test,
            )
        )
    return windows


def _contiguous_groups(n_obs: int, n_splits: int) -> List[np.ndarray]:
    bounds = np.linspace(0, n_obs, n_splits + 1).astype(np.int64)
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]


def purged_kfold(
    n_obs: int,
    n_splits: int,
    embargo: int = 0,
) -> List[Window]:
    """
    Purged K-fold over contiguous blocks of the series.

    Unlike scikit-learn's KFold this never shuffles: each fold is a contiguous
    block, and training is the purged complement. Folds still overlap in time,
    so this is a diagnostic rather than a realistic live simulation — use
    :func:`purged_walk_forward` for that.
    """
    if n_splits < 2:
        raise ValueError(f'n_splits must be at least 2, got {n_splits}')
    if n_obs < n_splits:
        raise ValueError(f'n_obs ({n_obs}) cannot cover {n_splits} splits')

    groups = _contiguous_groups(n_obs, n_splits)
    windows: List[Window] = []
    for index, test in enumerate(groups):
        if test.size == 0:
            continue
        mask = np.ones(n_obs, dtype=bool)
        mask[test] = False
        candidate = np.flatnonzero(mask)
        windows.append(
            Window(
                index=index,
                train=purge_train_indices(candidate, test, embargo),
                test=test,
            )
        )
    return windows


def combinatorial_paths(n_splits: int, n_test_splits: int) -> List[tuple]:
    """
    Every choice of ``n_test_splits`` blocks out of ``n_splits``.

    The count is C(n_splits, n_test_splits), which grows fast: 6 choose 2 is 15
    paths, 10 choose 5 is 252.
    """
    if n_test_splits >= n_splits:
        raise ValueError(
            f'n_test_splits ({n_test_splits}) must be smaller than '
            f'n_splits ({n_splits})'
        )
    return list(combinations(range(n_splits), n_test_splits))


def combinatorial_purged_split(
    n_obs: int,
    n_splits: int,
    n_test_splits: int,
    embargo: int = 0,
) -> List[Window]:
    """
    Combinatorial purged cross-validation.

    Splits the series into ``n_splits`` contiguous blocks, then for every
    combination of ``n_test_splits`` blocks builds one path: those blocks are
    the test set (purged together, so overlapping test blocks do not poison
    each other), and the purged remainder is the training set.

    A strategy therefore gets scored along many routes through the same data,
    which exposes results that depend on one lucky partition.
    """
    if n_obs < n_splits:
        raise ValueError(f'n_obs ({n_obs}) cannot cover {n_splits} splits')

    groups = _contiguous_groups(n_obs, n_splits)
    windows: List[Window] = []

    for index, path in enumerate(combinatorial_paths(n_splits, n_test_splits)):
        test = np.concatenate([groups[i] for i in path])
        test.sort()
        mask = np.ones(n_obs, dtype=bool)
        mask[test] = False
        candidate = np.flatnonzero(mask)
        windows.append(
            Window(
                index=index,
                train=purge_train_indices(candidate, test, embargo),
                test=test,
            )
        )
    return windows