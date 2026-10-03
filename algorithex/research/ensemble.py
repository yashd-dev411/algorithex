"""
Combining strategies: which ones to believe, and when.

Picking one strategy is the wrong frame. You usually have several that each
work part of the time, and the question is not "which is best" but "how much
of each, given what I know right now". Answering it well is worth more than
any additional indicator, because two mediocre strategies with uncorrelated
returns beat one good strategy that got unlucky.

Four things this module refuses to do:

It does not let one strategy dominate by accident
    Weights are normalised and can be floored, so a single spectacular month
    cannot silently take the whole book. :func:`effective_strategies` reports
    the Herfindahl inverse so you can see when the ensemble has quietly
    collapsed into being one strategy.

It does not peek
    :func:`walk_forward_ensemble` refits on a trailing window only. A weight
    computed from the future is not a weight, it is a label.

It does not fit one weight per regime on thin evidence
    :func:`regime_arbitration` shrinks each regime's allocation towards the
    global one, and refuses to specialise below a minimum observation count.
    Per-regime stacking on 30 bars of data is how people invent alphas.

It does not quietly drop members
    :func:`marginal_contribution` measures what each member is actually
    contributing, so removal is a decision you made rather than one that
    happened to you.

Everything takes a per-bar return matrix aligned to a shared index space, so
it composes with the trading engine without knowing anything about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'StrategyReturns',
    'WeightPath',
    'EnsembleReport',
    'build_returns_matrix',
    'positions_to_returns',
    'equal_weight_weights',
    'inverse_volatility_weights',
    'exponential_weights',
    'drawdown_penalised_weights',
    'information_ratio_weights',
    'ensemble_returns',
    'effective_strategies',
    'max_drawdown',
    'weight_stability',
    'regime_arbitration',
    'apply_regime_allocation',
    'walk_forward_ensemble',
    'marginal_contribution',
    'evaluate_ensemble',
]


@dataclass(frozen=True)
class StrategyReturns:
    """One strategy's per-bar return series."""

    name: str
    returns: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.returns, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f'strategy {self.name!r} returns must be one dimensional, '
                f'got shape {values.shape}'
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f'strategy {self.name!r} has non-finite returns')
        object.__setattr__(self, 'returns', values)


@dataclass
class WeightPath:
    """A time series of allocations, one row per bar."""

    matrix: np.ndarray
    names: List[str]
    scheme: str = 'custom'
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim == 1:
            # A lone allocation is a path of one bar, not a flattened path.
            matrix = (
                matrix.reshape(1, -1) if matrix.size else matrix.reshape(0, len(self.names))
            )
        if matrix.size and matrix.shape[1] != len(self.names):
            raise ValueError(
                f'weight matrix has {matrix.shape[1]} columns but '
                f'{len(self.names)} strategy names were given'
            )
        if matrix.size and not np.all(np.isfinite(matrix)):
            raise ValueError('weight matrix contains non-finite values')
        self.matrix = matrix

    @property
    def final(self) -> np.ndarray:
        """The allocation in force at the last bar."""
        if self.matrix.size == 0:
            raise ValueError('weight path is empty')
        return self.matrix[-1]

    def as_dict(self) -> Dict[str, float]:
        return {name: float(w) for name, w in zip(self.names, self.final)}


def build_returns_matrix(
    strategies: Sequence[StrategyReturns],
) -> Tuple[List[str], np.ndarray]:
    """
    Stack per-strategy return series into a ``(n_strategies, n_bars)`` matrix.

    Alignment is checked rather than assumed. Two strategies measured over
    different bar ranges are the single most common way an ensemble produces
    numbers that cannot be reproduced later.
    """
    if not strategies:
        raise ValueError('need at least one strategy')

    names: List[str] = []
    rows: List[np.ndarray] = []
    length: Optional[int] = None
    for strategy in strategies:
        if not isinstance(strategy, StrategyReturns):
            raise TypeError(f'expected StrategyReturns, got {type(strategy).__name__}')
        if length is None:
            length = strategy.returns.size
        elif strategy.returns.size != length:
            raise ValueError(
                f'strategy {strategy.name!r} has {strategy.returns.size} bars but '
                f'{names[0]!r} has {length}; the series must be aligned'
            )
        if strategy.name in names:
            raise ValueError(f'duplicate strategy name {strategy.name!r}')
        names.append(strategy.name)
        rows.append(strategy.returns)

    return names, np.vstack(rows)


def positions_to_returns(positions, prices, fee: float = 0.0) -> np.ndarray:
    """
    Turn a target-position stream into per-bar returns.

    `positions[t]` is the position held over ``[t, t+1]``; `prices[t]` to
    `prices[t+1]` is the move it captures. Applying `fee` on every change in
    position is what stops a high-turnover strategy from looking free.
    """
    held = np.asarray(positions, dtype=np.float64)
    marks = np.asarray(prices, dtype=np.float64)
    if held.ndim != 1 or marks.ndim != 1:
        raise ValueError('positions and prices must both be one dimensional')
    if held.size != marks.size:
        raise ValueError(
            f'positions ({held.size}) and prices ({marks.size}) must be the same length'
        )
    if not np.all(np.isfinite(held)) or not np.all(np.isfinite(marks)):
        raise ValueError('positions and prices must all be finite')
    if held.size < 2:
        raise ValueError(
            f'need at least 2 bars to compute a return, got {held.size}'
        )
    if np.any(marks == 0):
        raise ValueError('prices must be non-zero to compute returns')

    moves = marks[1:] / marks[:-1] - 1.0
    # The trade that puts `held[t]` on the book is charged to the period it is
    # held for, so the first bar pays the entry and flat periods pay nothing.
    traded = np.abs(np.diff(held, prepend=0.0))[: moves.size]
    return held[:-1] * moves - fee * traded


def _normalise(weights: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Scale to sum to one, optionally reserving `floor` for every member."""
    values = np.asarray(weights, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if floor > 0:
        values = floor + values
    total = values.sum()
    if total <= 0:
        raise ValueError('weights sum to zero; nothing to allocate')
    return values / total


def equal_weight_weights(n_strategies: int) -> np.ndarray:
    """One ``n``-th of capital each. The honest starting point."""
    if n_strategies < 1:
        raise ValueError(f'need at least one strategy, got {n_strategies}')
    return np.full(n_strategies, 1.0 / n_strategies)


def inverse_volatility_weights(
    returns_matrix: np.ndarray,
    floor_volatility: float = 1e-12,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Allocate inversely to each strategy's volatility.

    Volatility rather than return because return estimates are far noisier: a
    capitalisation on realised returns over a few hundred bars mostly
    capitalises on which strategy happened to be luckiest.

    A strategy with literally zero volatility still gets a finite share via
    `floor_volatility`; leaving it at zero would silently delete it.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.size == 0:
        raise ValueError('returns matrix is empty')
    volatilities = matrix.std(axis=1, ddof=1 if matrix.shape[1] > 1 else 0)
    volatilities = np.maximum(volatilities, floor_volatility)
    weights = 1.0 / volatilities
    weights = _normalise(weights)

    if max_weight is not None:
        if not 0 < max_weight <= 1:
            raise ValueError(f'max_weight must be in (0, 1], got {max_weight}')
        weights = _cap_weights(weights, max_weight)
    return weights


def _cap_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """
    Clamp weights at `max_weight`, giving the excess to whoever has the most
    headroom, repeating until nothing is left to redistribute.

    A plain clamp-and-renormalise loop oscillates forever when every weight is
    at the cap, so the redistribution target is chosen to have slack.
    """
    values = np.array(weights, dtype=np.float64, copy=True)
    n = values.size
    if max_weight * n < 1.0:
        raise ValueError(
            f'max_weight {max_weight} cannot be honoured across {n} members; '
            f'it needs to be at least {1.0 / n:.4f}'
        )
    for _ in range(n * 4):
        over = values > max_weight + 1e-15
        if not over.any():
            break
        excess = float((values[over] - max_weight).sum())
        values[over] = max_weight
        room = max_weight - values
        eligible = room > 1e-15
        if not eligible.any():
            break
        values[eligible] += excess * room[eligible] / room[eligible].sum()
    return values


def exponential_weights(
    returns_matrix: np.ndarray,
    learning_rate: float = 0.5,
    floor: float = 1e-6,
) -> np.ndarray:
    """
    Multiplicative-weights allocation that up-weights whatever just earned.

    Start equal, then after each bar multiply each member's weight by
    ``exp(learning_rate * bar_return)`` and renormalise. With `learning_rate`
    0 this is equal weight; larger values chase recent performance harder,
    which also means more overfitting to whichever member got lucky.

    The guarantee is real: for any fixed best member, the algorithm's
    cumulative weight beats it by at most ``log(n) / learning_rate``. That is
    why it beats a static allocation without needing to know the best member
    in advance.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.size == 0:
        raise ValueError('returns matrix is empty')
    if learning_rate < 0:
        raise ValueError(f'learning_rate must be non-negative, got {learning_rate}')
    if not np.isfinite(learning_rate):
        raise ValueError('learning_rate must be finite')

    n_strategies, n_bars = matrix.shape
    weights = np.full(n_strategies, 1.0 / n_strategies)
    for bar in range(n_bars):
        # Clip before exponentiating: one -900% bar would otherwise produce a
        # single weight of exactly zero and the member can never recover.
        bar_returns = np.clip(matrix[:, bar], -50.0, 50.0)
        weights = _normalise(weights * np.exp(learning_rate * bar_returns), floor=floor)
    return weights


def drawdown_penalised_weights(
    returns_matrix: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Inverse-volatility allocation scaled down by each member's worst fall.

    A member that halved is not a member you size like one that never moved,
    even if both have the same volatility. Weight goes as
    ``1 / vol / (1 + strength * max_drawdown)``.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.size == 0:
        raise ValueError('returns matrix is empty')
    if strength < 0:
        raise ValueError(f'strength must be non-negative, got {strength}')

    volatilities = np.maximum(matrix.std(axis=1, ddof=1 if matrix.shape[1] > 1 else 0), 1e-12)
    penalties = np.array([max_drawdown(row) for row in matrix])
    return _normalise(1.0 / volatilities / (1.0 + strength * penalties))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Deepest peak-to-trough fall of a return series, as a positive fraction.

    Computed on the equity curve rather than on the worst bar, because a
    strategy can avoid any single catastrophic bar and still bleed to death
    over twenty ordinary ones.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def ensemble_returns(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Combine member returns under an allocation.

    `weights` is either a single allocation applied to every bar, or a
    ``(n_bars, n_strategies)`` path where each bar uses its own allocation.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    allocation = np.asarray(weights, dtype=np.float64)
    if allocation.ndim == 1:
        if allocation.size != matrix.shape[0]:
            raise ValueError(
                f'weights has {allocation.size} entries but the matrix has '
                f'{matrix.shape[0]} strategies'
            )
        return allocation @ matrix
    if allocation.ndim != 2:
        raise ValueError(f'weights must be 1-D or 2-D, got shape {allocation.shape}')
    if allocation.shape[1] != matrix.shape[0]:
        raise ValueError(
            f'weight path has {allocation.shape[1]} columns but the matrix has '
            f'{matrix.shape[0]} strategies'
        )
    n_bars = min(allocation.shape[0], matrix.shape[1])
    if allocation.shape[0] != matrix.shape[1]:
        raise ValueError(
            f'weight path covers {allocation.shape[0]} bars but the matrix has '
            f'{matrix.shape[1]}'
        )
    return np.einsum('ts,st->t', allocation[:n_bars], matrix[:, :n_bars])


def information_ratio_weights(
    returns_matrix: np.ndarray,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Allocate in proportion to each member's information ratio.

    ``mean / volatility``, with losers scoring zero rather than negative --
    a member that lost money in this sample earns no capital here, which is
    the whole point of arbitrating between strategies on their *conditional*
    performance.

    Note this is deliberately different from :func:`inverse_volatility_weights`.
    Sizing on volatility alone cannot distinguish a member that earned from one
    that did not, so it can never arbitrate; it only equalises risk.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.size == 0:
        raise ValueError('returns matrix is empty')
    if matrix.shape[1] < 2:
        raise ValueError('need at least 2 bars to estimate a volatility')

    volatilities = np.maximum(
        matrix.std(axis=1, ddof=1), np.finfo(np.float64).tiny
    )
    scores = matrix.mean(axis=1) / volatilities
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    if scores.sum() <= 0:
        return equal_weight_weights(matrix.shape[0])
    return _normalise(scores, floor=floor)


def effective_strategies(weights: np.ndarray) -> float:
    """
    How many strategies an allocation behaves like, on a scale of 1 to n.

    The Herfindahl inverse: three equal weights give 3.0, and a 0.9/0.05/0.05
    split gives about 1.16. It is the number to watch when a "diversified
    ensemble" turns out to be one trade with extra steps.
    """
    allocation = np.asarray(weights, dtype=np.float64)
    if allocation.ndim != 1 or allocation.size == 0:
        raise ValueError('weights must be a non-empty 1-D array')
    total = allocation.sum()
    if total <= 0:
        raise ValueError('weights must sum to a positive number')
    normalised = allocation / total
    concentration = float(np.sum(normalised**2))
    return float(1.0 / concentration) if concentration > 0 else 0.0



def regime_arbitration(
    returns_matrix: np.ndarray,
    labels,
    shrinkage: float = 0.5,
    min_obs: int = 30,
) -> Tuple[Dict[object, np.ndarray], np.ndarray]:
    """
    Learn a separate allocation per regime, shrunk towards the global one.

    The premise is real: a trend system and a mean-reversion system earn in
    different conditions, and a single blended weight is a compromise that is
    wrong in both. The failure mode is equally real -- fitting one weight per
    regime on a handful of bars manufactures structure that is not there.

    So each regime's allocation is pulled `shrinkage` of the way back towards
    the global inverse-volatility weights, and any regime with fewer than
    `min_obs` bars gets the global weights untouched. The returned dictionary
    is therefore always a *damped* specialisation, never a free one.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    tags = np.asarray(labels)
    if tags.ndim != 1 or tags.size != matrix.shape[1]:
        raise ValueError(
            f'labels must be one dimensional and match the {matrix.shape[1]} bars, '
            f'got shape {tags.shape}'
        )
    if not 0 <= shrinkage <= 1:
        raise ValueError(f'shrinkage must be in [0, 1], got {shrinkage}')
    if min_obs < 1:
        raise ValueError(f'min_obs must be at least 1, got {min_obs}')

    global_weights = inverse_volatility_weights(matrix)
    n_strategies = matrix.shape[0]
    allocations: Dict[object, np.ndarray] = {}
    used = np.zeros(n_strategies, dtype=np.int64)

    for tag in np.unique(tags):
        mask = tags == tag
        observed = int(mask.sum())
        # Fewer than min_obs bars means not enough evidence to specialise, and
        # fewer than 2 means not even a volatility to divide by.
        if observed < min_obs or observed < 2:
            allocations[tag] = global_weights
            continue
        local = information_ratio_weights(matrix[:, mask])
        allocations[tag] = _normalise(shrinkage * global_weights + (1.0 - shrinkage) * local)
        used += observed

    return allocations, used


def apply_regime_allocation(
    allocations: Dict[object, np.ndarray],
    labels,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Expand a per-regime allocation table into a per-bar weight path.

    Labels with no learned allocation fall back to `fallback`, or to equal
    weight when none is supplied -- so an unseen regime degrades to something
    harmless instead of raising halfway through a backtest.
    """
    tags = np.asarray(labels)
    if tags.ndim != 1:
        raise ValueError(f'labels must be one dimensional, got shape {tags.shape}')
    if not allocations:
        raise ValueError('allocations must not be empty')

    n_strategies = len(next(iter(allocations.values())))
    default = (
        np.asarray(fallback, dtype=np.float64)
        if fallback is not None
        else equal_weight_weights(n_strategies)
    )
    if default.size != n_strategies:
        raise ValueError(
            f'fallback has {default.size} entries but allocations have {n_strategies}'
        )

    path = np.empty((tags.size, n_strategies), dtype=np.float64)
    unknown = np.zeros(tags.size, dtype=bool)
    for i, tag in enumerate(tags):
        allocation = allocations.get(tag)
        if allocation is None:
            path[i] = default
            unknown[i] = True
        else:
            path[i] = allocation
    return path, unknown


# --- causal refitting -----------------------------------------------------


def walk_forward_ensemble(
    returns_matrix: np.ndarray,
    scheme: str = 'inverse_volatility',
    train_window: int = 252,
    **kwargs,
) -> WeightPath:
    """
    Refit the allocation on a trailing window and rebalance each bar.

    This is the difference between a backtest you can trust and one you
    cannot: at bar ``t`` the weights come only from bars ``< t``. The
    allocation is therefore biased towards equal weight until the first window
    fills, which is honest -- there is genuinely no evidence before then.

    `scheme` is one of ``equal_weight``, ``inverse_volatility``,
    ``exponential``, ``drawdown`` or ``regime``; the last needs `labels`.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if matrix.size == 0:
        raise ValueError('returns matrix is empty')
    if train_window < 2:
        raise ValueError(f'train_window must be at least 2, got {train_window}')
    n_strategies, n_bars = matrix.shape

    labels = kwargs.pop('labels', None)
    if scheme == 'regime':
        if labels is None:
            raise ValueError("scheme 'regime' needs a labels argument")

    def fit(history: np.ndarray, start: int, upto: int) -> np.ndarray:
        if history.shape[1] == 0:
            return equal_weight_weights(n_strategies)
        if scheme == 'equal_weight':
            return equal_weight_weights(n_strategies)
        if scheme == 'inverse_volatility':
            return inverse_volatility_weights(history, max_weight=kwargs.get('max_weight'))
        if scheme == 'exponential':
            return exponential_weights(history, learning_rate=kwargs.get('learning_rate', 0.5))
        if scheme == 'drawdown':
            return drawdown_penalised_weights(history, strength=kwargs.get('strength', 1.0))
        if scheme == 'regime':
            # Slice the labels with the same window as the returns, or the two
            # index spaces drift apart and stop describing the same bars.
            allocations, _ = regime_arbitration(
                history,
                np.asarray(labels)[start:upto],
                shrinkage=kwargs.get('shrinkage', 0.5),
                min_obs=kwargs.get('min_obs', 30),
            )
            current = np.asarray(labels)[upto]
            return allocations.get(current, inverse_volatility_weights(history))
        raise ValueError(f'unknown allocation scheme {scheme!r}')

    path = np.empty((n_bars, n_strategies), dtype=np.float64)
    for t in range(n_bars):
        start = max(0, t - train_window)
        path[t] = fit(matrix[:, start:t], start, t)

    return WeightPath(
        matrix=path,
        names=[f'strategy_{i}' for i in range(n_strategies)],
        scheme=scheme,
        meta={'train_window': int(train_window), 'bars': int(n_bars)},
    )


# --- reporting ------------------------------------------------------------


def marginal_contribution(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    periods_per_year: int = 365,
) -> List[Dict[str, float]]:
    """
    What each member adds to the ensemble, by removal.

    For every member, drop it, redistribute its weight across the rest in
    proportion, and re-score. A member with a negative contribution is not
    merely useless -- it is costing the ensemble, and that is invisible in a
    list of per-strategy Sharpe ratios.
    """
    matrix = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    allocation = _normalise(np.asarray(weights, dtype=np.float64))
    n_strategies = matrix.shape[0]
    full = performance_metrics(ensemble_returns(matrix, allocation), periods_per_year)

    contributions: List[Dict[str, float]] = []
    for i in range(n_strategies):
        survivors = [j for j in range(n_strategies) if j != i]
        if not survivors:
            reduced = np.zeros(matrix.shape[1])
        else:
            reduced_weights = allocation[survivors]
            reduced = ensemble_returns(matrix[survivors], reduced_weights)
        without = performance_metrics(reduced, periods_per_year)
        full_sharpe = full.get('sharpe', float('nan'))
        lean_sharpe = without.get('sharpe', float('nan'))
        contributions.append(
            {
                'index': i,
                'weight': float(allocation[i]),
                'sharpe_with': float(full_sharpe),
                'sharpe_without': float(lean_sharpe),
                'delta_sharpe': float(full_sharpe - lean_sharpe),
                'return_with': float(full.get('total_return', float('nan'))),
                'return_without': float(without.get('total_return', float('nan'))),
            }
        )
    return contributions


@dataclass
class EnsembleReport:
    """A side-by-side of the ensemble and every member it was built from."""

    names: List[str]
    weights: np.ndarray
    ensemble: Dict[str, float]
    members: Dict[str, Dict[str, float]]
    contributions: List[Dict[str, float]]
    weight_stability: float
    effective_strategies: float
    meta: dict = field(default_factory=dict)

    def best_member(self) -> str:
        """The strongest single strategy on its own, which is not the point."""
        if not self.members:
            raise ValueError('no members to compare')
        return max(
            self.members, key=lambda name: self.members[name].get('sharpe', float('-inf'))
        )

    def beats_best_member(self) -> bool:
        """Whether combining actually added anything over the single best."""
        best = self.best_member()
        return self.ensemble.get('sharpe', float('-inf')) > self.members[best].get(
            'sharpe', float('-inf')
        )

    def harmful_members(self) -> List[str]:
        """Members whose removal would improve the ensemble's Sharpe."""
        by_index = {c['index']: name for name, c in zip(self.names, self.contributions)}
        return [
            by_index[c['index']] for c in self.contributions if c['delta_sharpe'] < 0
        ]

    def as_dict(self) -> dict:
        return {
            'names': list(self.names),
            'weights': {n: round(float(w), 6) for n, w in zip(self.names, self.weights)},
            'ensemble': {k: round(float(v), 6) for k, v in self.ensemble.items()},
            'members': {
                n: {k: round(float(v), 6) for k, v in m.items()}
                for n, m in self.members.items()
            },
            'weight_stability': round(self.weight_stability, 6),
            'effective_strategies': round(self.effective_strategies, 6),
        }


def weight_stability(path: np.ndarray) -> float:
    """
    Mean absolute bar-to-bar change in the allocation, scaled by ``n``.

    0 means the weights never moved; 1 means a member went from all of the
    book to none between consecutive bars. An unstable allocator is a
    turnover generator, and turnover is a cost the Sharpe ratio does not show.
    """
    matrix = np.asarray(path, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    changes = np.abs(np.diff(matrix, axis=0))
    return float(changes.mean() / max(matrix.shape[1], 1))


def evaluate_ensemble(
    strategies: Sequence[StrategyReturns],
    weights: Optional[np.ndarray] = None,
    weight_path: Optional[np.ndarray] = None,
    periods_per_year: int = 365,
    max_weight: Optional[float] = None,
) -> EnsembleReport:
    """
    Score an ensemble against each of its members.

    Pass `weight_path` for an honest walk-forward evaluation, or `weights` for
    a single fixed allocation. With neither, equal weight is used -- which is
    the right default, because picking weights on the same data you score
    them on is the most common way to manufacture an ensemble that looks good
    and is not.
    """
    names, matrix = build_returns_matrix(strategies)

    if weight_path is not None:
        allocation = np.asarray(weight_path, dtype=np.float64)
        if allocation.ndim == 1:
            allocation = np.tile(allocation, (matrix.shape[1], 1))
        combined = ensemble_returns(matrix, allocation)
        final = allocation[-1]
        stability = weight_stability(allocation)
    else:
        if weights is not None:
            final = inverse_volatility_weights(matrix, max_weight=max_weight) if weights is None else _normalise(weights)
        else:
            final = inverse_volatility_weights(matrix, max_weight=max_weight)
        combined = ensemble_returns(matrix, final)
        stability = 0.0

    members = {
        name: performance_metrics(row, periods_per_year) for name, row in zip(names, matrix)
    }
    return EnsembleReport(
        names=names,
        weights=final,
        ensemble=performance_metrics(combined, periods_per_year),
        members=members,
        contributions=marginal_contribution(matrix, final, periods_per_year),
        weight_stability=stability,
        effective_strategies=effective_strategies(final),
        meta={'weight_source': 'path' if weight_path is not None else 'static'},
    )


def performance_metrics(returns, periods_per_year: int = 365) -> Dict[str, float]:
    """
    Per-bar return statistics, re-exported from the walk-forward package so
    this module scores exactly what the validation code scores.
    """
    from .walkforward.evaluate import performance_metrics as _metrics

    return _metrics(returns, periods_per_year)
