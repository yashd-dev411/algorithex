"""
Change-point detection.

A hidden Markov model answers "which regime am I in"; change-point detection
answers "when did the regime change". For sizing and risk limits the second
question is often the more useful one, because it tells you how much history
behind you actually belongs to the current environment.

The method is binary segmentation with a BIC penalty. It is not the most
powerful algorithm available ( PELT is faster, Bayesian alternatives give
posteriors) but it is exact up to the greedy split choice, needs no tuning
parameters beyond a penalty, and is easy to read and to test against a series
with known breakpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ['ChangePoint', 'segment_cost', 'best_split', 'detect_change_points', 'segments']

# Smallest positive normal float. Used as the variance floor so that perfectly
# flat segments stay describable at a finite cost.
_DEFAULT_FLOOR = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True)
class ChangePoint:
    """A detected breakpoint at position ``index``, meaning bars before and
    after it are described by different distributions."""

    index: int
    score: float

    def __repr__(self) -> str:
        return f'ChangePoint(index={self.index}, score={self.score:.2f})'


def segment_cost(values: np.ndarray, variance_floor: float = _DEFAULT_FLOOR) -> float:
    """
    Cost of describing a segment with one constant Gaussian.

    Lower is better, and this is the objective the segmentation minimises. A
    constant segment of length n with variance s costs ``n * log(s)``.

    A perfectly flat segment has zero variance, which would make the cost
    ``-inf`` and let the search "win" by splitting off a couple of identical
    bars rather than finding the real level shift. The variance is therefore
    floored at the smallest positive float: that keeps every cost finite while
    still making flat segments strongly preferred over mixed ones.

    :param values: the segment
    :param variance_floor: smallest variance treated as achievable
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        # A single point has no variance to describe.
        return 0.0
    variance = max(float(np.var(values)), variance_floor)
    return n * np.log(variance)


def _cost_from_moments(total: float, total_sq: float, n: int,
                       variance_floor: float = _DEFAULT_FLOOR) -> float:
    """Cost of a segment described by its sum of values and sum of squares."""
    if n < 2:
        return 0.0
    variance = total_sq / n - (total / n) ** 2
    # Rounding can push a genuinely constant segment slightly below zero.
    variance = max(variance, variance_floor)
    return n * np.log(variance)


def best_split(values: np.ndarray, min_segment: int = 2,
               variance_floor: float = _DEFAULT_FLOOR):
    """
    The split point minimising total segment cost.

    Uses running sums of values and squares so each candidate costs O(1)
    rather than re-slicing the array for every possible split.

    :param values: the segment to split
    :param min_segment: smallest permitted segment on either side
    :return: ``(split_index, total_cost)`` or ``(None, inf)`` when no legal
        split exists
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size

    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(values))))

    best_index = None
    best_cost = np.inf

    for i in range(min_segment, n - min_segment + 1):
        left = _cost_from_moments(prefix[i], prefix_sq[i], i, variance_floor)
        right = _cost_from_moments(
            prefix[n] - prefix[i], prefix_sq[n] - prefix_sq[i], n - i, variance_floor
        )
        total = left + right
        if total < best_cost:
            best_cost = total
            best_index = i

    return best_index, best_cost


def _penalty(n: int, penalty_scale: float) -> float:
    """
    Cost of accepting one breakpoint.

    The segment cost is ``n * log(variance)``, so the saving from a genuine
    level shift is O(n). The penalty therefore has to scale like O(log n) —
    one model parameter per breakpoint — or it dwarfs every real gain and no
    change point is ever found.
    """
    return penalty_scale * np.log(max(n, 2))


def detect_change_points(
    values,
    min_segment: int = 10,
    n_points: Optional[int] = None,
    penalty_scale: float = 3.0,
) -> List[ChangePoint]:
    """
    Find change points by greedy binary segmentation.

    At each step the whole current segment is searched for the split that best
    explains it as two homogeneous pieces; the best breakpoint is accepted
    only if it pays for itself against the BIC-style penalty, then the two
    halves are searched independently.

    :param values: 1D series
    :param min_segment: shortest run of bars kept on either side of a split
    :param n_points: cap on how many breakpoints to report; the most
        significant are kept
    :param penalty_scale: cost of a breakpoint in units of the BIC penalty.
        Higher finds fewer, more conservative breakpoints. The default of 3
        recovers clean level shifts in synthetic series while reporting none on
        stationary noise; values near 1 start splitting gradual drift into
        spurious steps
    :return: change points ordered by position
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f'expected a 1D series, got shape {series.shape}')
    # Validate before the length short-circuit: corrupt input should be
    # reported even when the series is too short to analyse anyway.
    if not np.all(np.isfinite(series)):
        raise ValueError('series contains NaN or infinite values')
    if min_segment < 1:
        raise ValueError(f'min_segment must be at least 1, got {min_segment}')
    if series.size < 2 * min_segment:
        return []

    penalty = _penalty(series.size, penalty_scale)
    found: List[ChangePoint] = []
    # Stack of (start, end) segments still to examine.
    stack: List[tuple] = [(0, series.size)]

    while stack:
        start, end = stack.pop()
        segment = series[start:end]
        if segment.size < 2 * min_segment:
            continue

        split, cost = best_split(segment, min_segment)
        if split is None:
            continue

        gain = segment_cost(segment) - cost - penalty
        if not np.isfinite(gain) or gain <= 0:
            continue

        position = start + split
        found.append(ChangePoint(index=position, score=float(gain)))
        stack.append((start, position))
        stack.append((position, end))

    found.sort(key=lambda cp: (-cp.score, cp.index))
    if n_points is not None:
        found = found[:n_points]
    found.sort(key=lambda cp: cp.index)
    return found


def segments(values, change_points: List[ChangePoint]):
    """
    Split a series into the runs implied by detected change points.

    :return: list of ``(start, end)`` half-open index pairs, in order
    """
    series = np.asarray(values)
    bounds = [0] + [cp.index for cp in sorted(change_points, key=lambda c: c.index)]
    bounds.append(series.size)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]