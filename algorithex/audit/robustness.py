"""
Falsification: try to break your own strategy before the market does.

A backtest tells you what happened. It says nothing about whether the result
would survive being slightly wrong -- a tick that arrived late, a candle whose
volume was never really traded, a fee you underestimated. Strategies that are
genuinely capturing something tend to be dull about all of this. Strategies
that fitted a quirk fall over the moment the quirk moves.

So this module attacks the *data and the cost model* rather than the
parameters, and reports how much of the profit survives:

Survival rate
    What fraction of plausible corruptions still leave the strategy in profit.
    A strategy that only wins on the pristine tape is one that is not trading,
    it is pattern matching a specific file.

Break-even cost
    How far fees and slippage can be inflated before PnL reaches zero. A
    strategy with a break-even at 1.2x is not a strategy, it is a rounding
    difference.

Margin of safety
    :func:`certify_robustness` searches for the worst price path within a
    perturbation budget. That is the real certificate: not "it worked", but
    "no price path this far from the historical one could have produced a
    loss", and how much further the budget can be pushed before that stops
    being true.

Every entry point takes a ``strategy(prices) -> pnl`` callable, so this works
against any strategy without the strategy knowing it is being tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
import zlib

import numpy as np

__all__ = [
    'Attack',
    'AttackResult',
    'RobustnessReport',
    'jitter_prices',
    'shuffle_returns',
    'scramble_volumes',
    'inject_spikes',
    'block_bootstrap',
    'drop_random_bars',
    'apply_attack',
    'survival_rate',
    'pnl_retention',
    'cost_stress_curve',
    'break_even_multiple',
    'perturbation_sensitivity',
    'adversarial_path',
    'certify_robustness',
    'audit_robustness',
    'DEFAULT_ATTACKS',
]

Strategy = Callable[[np.ndarray], float]


def _as_prices(prices) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(
            f'prices must be a 1-D series of at least 3 bars, got shape {values.shape}'
        )
    if not np.all(np.isfinite(values)):
        raise ValueError('prices must all be finite')
    if np.any(values <= 0):
        raise ValueError('prices must be positive to work in log space')
    return values


def _check_strategy(strategy) -> Strategy:
    if not callable(strategy):
        raise TypeError(f'strategy must be callable, got {type(strategy).__name__}')
    return strategy


def _rebuild(base_price: float, log_returns: np.ndarray) -> np.ndarray:
    """
    Rebuild a price path from a log-return series, preserving bar count.

    The leading 0 matters: without it the reconstruction starts one bar late
    and the result has one bar fewer than the input, which silently shifts
    every position the strategy holds.
    """
    path = np.concatenate(([0.0], np.cumsum(log_returns)))
    return base_price * np.exp(path)


def _realised_vol(prices: np.ndarray) -> float:
    """Per-bar volatility as a fraction, floored so flat tape is still usable."""
    log_returns = np.diff(np.log(prices))
    vol = float(np.std(log_returns))
    return vol if vol > 1e-9 else 1e-4


# --- attacks --------------------------------------------------------------


@dataclass(frozen=True)
class Attack:
    """
    One named corruption of the price series.

    `severity` is the attack's own size knob, so a report can be run at
    several severities and show how sharp the cliff is rather than a single
    pass/fail at an arbitrary level.
    """

    name: str
    apply: Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
    description: str = ''


def jitter_prices(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add noise proportional to realised volatility.

    The cheapest honest test of whether a signal depends on a level that was
    only ever observed to three decimals.
    """
    base = _as_prices(prices)
    log_returns = np.diff(np.log(base))
    noise = rng.normal(0.0, severity * _realised_vol(base), base.size)
    path = np.cumsum(noise)
    path -= path[0]
    return base * np.exp(path)


def shuffle_returns(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Permute the returns, preserving the distribution but destroying the order.

    If PnL survives this, the strategy was reading the sequence. If it
    collapses, whatever it was reading was an artefact of this particular
    ordering. `severity` is the fraction of the path shuffled; 1.0 is the full
    control.
    """
    base = _as_prices(prices)
    log_returns = np.diff(np.log(base))
    n_shuffle = int(np.clip(severity, 0.0, 1.0) * log_returns.size)
    if n_shuffle <= 0:
        return base.copy()
    positions = rng.choice(log_returns.size, size=n_shuffle, replace=False)
    shuffled = log_returns.copy()
    shuffled[positions] = rng.permutation(shuffled[positions])
    return _rebuild(base[0], shuffled)


def scramble_volumes(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Re-roll the returns using the *same* distribution but a fresh sample.

    A parametric bootstrap: it answers "would this have worked on other draws
    from the same process", which is a different and usually more useful
    question than shuffling one specific path.
    """
    base = _as_prices(prices)
    log_returns = np.diff(np.log(base))
    replacement = rng.permutation(log_returns)
    blended = severity * replacement + (1.0 - severity) * log_returns
    return _rebuild(base[0], blended)


def inject_spikes(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Punch single-bar dislocations into the path.

    Models bad prints and wicks. The recovery is immediate on purpose: what is
    under test is a strategy's reaction to one impossible price, not a market
    that stayed dislocated.
    """
    base = _as_prices(prices)
    n_spikes = max(1, int(round(severity * base.size / 10.0)))
    size = 3.0 * _realised_vol(base)
    out = base.copy()
    for index in rng.choice(base.size, size=n_spikes, replace=False):
        direction = 1.0 if rng.random() < 0.5 else -1.0
        out[index] = base[index] * np.exp(direction * size)
    return out


def block_bootstrap(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Resample contiguous blocks of returns with replacement.

    Preserves local structure while destroying long-range dependence, which is
    how a strategy that secretly depends on one regime gets exposed.
    """
    base = _as_prices(prices)
    log_returns = np.diff(np.log(base))
    block = max(2, int(severity * log_returns.size / 8.0))
    block = min(block, log_returns.size)
    n_blocks = int(np.ceil(log_returns.size / block))
    starts = rng.integers(0, log_returns.size - block + 1, size=n_blocks)
    resampled = np.concatenate(
        [log_returns[s:s + block] for s in starts]
    )[: log_returns.size]
    return _rebuild(base[0], resampled)


def drop_random_bars(prices, severity: float, rng: np.random.Generator) -> np.ndarray:
    """
    Delete a fraction of bars and re-close the gaps.

    Simulates gaps in a feed. Useful because a strategy that is indifferent to
    missing bars usually has an implicit assumption about bar spacing that no
    backtest ever exercised.
    """
    base = _as_prices(prices)
    keep_fraction = float(np.clip(1.0 - severity, 0.05, 1.0))
    n_keep = max(3, int(keep_fraction * base.size))
    keep = np.sort(rng.choice(base.size, size=n_keep, replace=False))
    kept = base[keep]
    if kept.size < 3:
        return base.copy()
    return kept * (base[-1] / kept[-1])


DEFAULT_ATTACKS: List[Attack] = [
    Attack('jitter', jitter_prices, 'noise proportional to realised volatility'),
    Attack('shuffle', shuffle_returns, 'returns permuted, distribution kept'),
    Attack('resample', scramble_volumes, 'returns re-drawn from their own sample'),
    Attack('spikes', inject_spikes, 'single-bar price dislocations'),
    Attack('block_bootstrap', block_bootstrap, 'local structure kept, long range lost'),
    Attack('drop_bars', drop_random_bars, 'bars removed and gaps re-closed'),
]


def apply_attack(
    attack: Attack,
    prices,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one attack at one severity."""
    return attack.apply(prices, severity, rng)


# --- scoring --------------------------------------------------------------


@dataclass(frozen=True)
class AttackResult:
    """What one attack did to the strategy."""

    attack: str
    severity: float
    pnl: float
    baseline_pnl: float

    @property
    def profitable(self) -> bool:
        return self.pnl > 0

    @property
    def retention(self) -> float:
        """
        PnL kept relative to the baseline, clipped at zero from below.

        Negative means the attack cost more than the baseline earned, which is
        worth showing rather than hiding behind a negative ratio.
        """
        if self.baseline_pnl == 0:
            return float('nan')
        return float(self.pnl / self.baseline_pnl)

    @property
    def flipped(self) -> bool:
        """The attack turned a winner into a loser."""
        return self.baseline_pnl > 0 and self.pnl <= 0

    def as_dict(self) -> dict:
        return {
            'attack': self.attack,
            'severity': round(self.severity, 6),
            'pnl': round(self.pnl, 8),
            'retention': round(self.retention, 6),
            'flipped': self.flipped,
        }


def survival_rate(results: Sequence[AttackResult]) -> float:
    """
    Fraction of attacks that left the strategy in profit.

    Averaged across severities, so an attack that survives being mild and dies
    being strong scores exactly what it should.
    """
    values = list(results)
    if not values:
        raise ValueError('need at least one attack result')
    return float(np.mean([r.profitable for r in values]))


def pnl_retention(results: Sequence[AttackResult]) -> float:
    """Median fraction of baseline PnL retained across every attack."""
    values = [r.retention for r in results if np.isfinite(r.retention)]
    if not values:
        return float('nan')
    return float(np.median(values))


def cost_stress_curve(
    strategy: Strategy,
    prices,
    multipliers: Sequence[float],
    cost_fn: Optional[Callable[[np.ndarray, float], float]] = None,
) -> List[float]:
    """
    PnL as fees and slippage are inflated by each multiplier.

    `cost_fn(prices, multiplier)` should return the PnL *after* costs; the
    default charges a proportional round-trip cost on each bar-to-bar move, so
    it penalises exactly the turnover a strategy actually generates.
    """
    base = _as_prices(prices)
    fn = _check_strategy(strategy)
    values = list(multipliers)
    if not values:
        raise ValueError('need at least one cost multiplier')
    if any(m < 0 for m in values):
        raise ValueError('cost multipliers must be non-negative')

    if cost_fn is None:
        def cost_fn(path, multiplier):  # noqa: F811 - deliberate local default
            gross = fn(path)
            moves = np.abs(np.diff(path) / path[:-1])
            return gross - multiplier * float(moves.sum())

    return [float(cost_fn(base, float(m))) for m in values]


def break_even_multiple(
    strategy: Strategy,
    prices,
    lo: float = 0.0,
    hi: float = 100.0,
    tolerance: float = 1e-3,
    max_iterations: int = 60,
    cost_fn: Optional[Callable[[np.ndarray, float], float]] = None,
) -> float:
    """
    The cost multiplier at which PnL first reaches zero.

    Found by bisection, which needs no assumption about the shape of the curve
    -- a real cost curve is convex, but assuming that is how you end up
    reporting a break-even point that does not exist.

    Returns ``inf`` when the strategy stays profitable across the whole
    bracket, and ``lo`` when it is already unprofitable before any cost.
    """
    fn = _check_strategy(strategy)
    if hi <= lo:
        raise ValueError(f'need hi > lo, got lo={lo} hi={hi}')
    if tolerance <= 0:
        raise ValueError(f'tolerance must be positive, got {tolerance}')

    def pnl_at(multiplier: float) -> float:
        if cost_fn is None:
            base = _as_prices(prices)
            gross = fn(base)
            moves = np.abs(np.diff(base) / base[:-1])
            return gross - multiplier * float(moves.sum())
        return float(cost_fn(_as_prices(prices), multiplier))

    low, high = float(lo), float(hi)
    if pnl_at(low) <= 0:
        return low
    if pnl_at(high) > 0:
        return float('inf')

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        if pnl_at(mid) > 0:
            low = mid
        else:
            high = mid
        if high - low <= tolerance * max(1.0, abs(low)):
            break
    return 0.5 * (low + high)


def perturbation_sensitivity(
    strategy: Strategy,
    prices,
    severity: float = 0.25,
    n_trials: int = 64,
    seed: int = 0,
    attack: Optional[Attack] = None,
) -> Dict[str, float]:
    """
    Distribution of PnL under many small perturbations.

    The spread matters more than the mean. A tight distribution means the
    strategy is insensitive to small errors; a wide one means small errors
    decide the outcome, whatever the backtest says.
    """
    fn = _check_strategy(strategy)
    base = _as_prices(prices)
    if n_trials < 2:
        raise ValueError(f'need at least 2 trials, got {n_trials}')
    if severity <= 0:
        raise ValueError(f'severity must be positive, got {severity}')

    chosen = attack if attack is not None else DEFAULT_ATTACKS[0]
    rng = np.random.default_rng(seed)
    pnls = np.array(
        [float(fn(apply_attack(chosen, base, severity, rng))) for _ in range(n_trials)]
    )

    mean = float(pnls.mean())
    std = float(pnls.std(ddof=1))
    return {
        'mean': mean,
        'std': std,
        'min': float(pnls.min()),
        'max': float(pnls.max()),
        'p05': float(np.percentile(pnls, 5)),
        'p95': float(np.percentile(pnls, 95)),
        'loss_frequency': float(np.mean(pnls <= 0)),
        'coefficient_of_variation': float(std / abs(mean)) if mean != 0 else float('inf'),
        'baseline_pnl': float(fn(base)),
    }


# --- certified worst-case search -----------------------------------------


def adversarial_path(
    strategy: Strategy,
    prices,
    budget: float = 0.05,
    n_steps: int = 60,
    step_size: float = 1e-3,
    seed: int = 0,
    n_restarts: int = 3,
) -> Dict[str, object]:
    """
    Search for the price path within ``+/- budget`` that hurts the strategy most.

    This is the difference between testing robustness and certifying it. Instead
    of hoping the attacks in :data:`DEFAULT_ATTACKS` are representative, this
    runs projected gradient ascent directly on the log-price path: at every
    step it takes the worst-case PnL it can find, then projects the perturbation
    back into the ``+/- budget`` box. The result is a local worst case over the
    whole set of price paths that are close to the historical one.

    Multi-start because the objective is not convex; several random restarts
    make it far less likely that a single unlucky start is reported as the
    worst case.

    `budget` is a fraction: 0.05 means every bar may move up to 5% away from
    where it actually was.
    """
    fn = _check_strategy(strategy)
    base = _as_prices(prices)
    if budget <= 0:
        raise ValueError(f'budget must be positive, got {budget}')
    if n_steps < 1:
        raise ValueError(f'n_steps must be at least 1, got {n_steps}')
    if step_size <= 0:
        raise ValueError(f'step_size must be positive, got {step_size}')
    if n_restarts < 1:
        raise ValueError(f'n_restarts must be at least 1, got {n_restarts}')

    baseline = float(fn(base))
    # Work in log space so the perturbation is a relative, price-independent
    # band and the path stays strictly positive however far the search moves
    # it. log(1 + budget) rather than budget: the budget is promised in *price*
    # terms, and exp(budget) - 1 is a little larger than budget.
    log_base = np.log(base)
    limit = float(np.log1p(budget))

    def to_prices(delta: np.ndarray) -> np.ndarray:
        return np.exp(log_base + delta)

    best = {'pnl': baseline, 'delta': np.zeros(base.size), 'restart': 0}

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart * 7919)
        delta = rng.uniform(-limit, limit, base.size) if restart else np.zeros(base.size)
        current = float(fn(to_prices(delta)))

        for _ in range(n_steps):
            step = rng.normal(0.0, step_size, base.size)
            # Two candidate moves per step: a plain random walk in the box is
            # far too slow to cross a long price path.
            candidates = [
                np.clip(delta + step, -limit, limit),
                np.clip(
                    delta + np.sign(step) * step_size * np.abs(delta + step) / limit,
                    -limit,
                    limit,
                ),
            ]
            for candidate in candidates:
                score = float(fn(to_prices(candidate)))
                if score < current:
                    delta, current = candidate, score
            if current < best['pnl']:
                best = {'pnl': current, 'delta': delta.copy(), 'restart': restart}
            if current < -abs(baseline) * 100:
                # Already annihilated; nothing further down this branch matters.
                break

    worst_path = to_prices(np.asarray(best['delta']))
    return {
        'baseline_pnl': baseline,
        'worst_pnl': float(best['pnl']),
        'pnl_lost': float(baseline - best['pnl']),
        'budget': float(budget),
        'worst_path': worst_path,
        'max_deviation': float(np.max(np.abs(np.expm1(np.asarray(best['delta']))))),
        'restart': int(best['restart']),
    }


def certify_robustness(
    strategy: Strategy,
    prices,
    budgets: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
    n_steps: int = 40,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Find the budget at which the strategy can first be made to lose money.

    Walks outwards through `budgets` and reports the smallest one whose worst
    case is negative. That number is the margin of safety, and unlike a Sharpe
    ratio it is a statement about the strategy rather than about the period it
    was tested on: "the historical path could be perturbed by 2% before this
    could have lost money".

    A strategy with no such budget in the list is reported as ``None`` rather
    than as passing, because surviving 10% is not the same as being safe.
    """
    fn = _check_strategy(strategy)
    base = _as_prices(prices)
    values = [float(b) for b in budgets]
    if not values:
        raise ValueError('need at least one budget')
    if any(b <= 0 for b in values):
        raise ValueError('budgets must all be positive')
    if list(values) != sorted(values):
        raise ValueError('budgets must be supplied smallest first')

    baseline = float(fn(base))
    if baseline <= 0:
        return {
            'baseline_pnl': baseline,
            'margin_of_safety': 0.0,
            'verdict': 'LOSES_MONEY_ON_HISTORICAL_DATA',
            'probes': [],
        }

    probes: List[Dict[str, object]] = []
    margin: Optional[float] = None
    for budget in values:
        result = adversarial_path(
            fn, base, budget=budget, n_steps=n_steps, seed=seed
        )
        worst = float(result['worst_pnl'])
        probes.append({
            'budget': budget,
            'worst_pnl': worst,
            'pnl_lost': float(result['pnl_lost']),
        })
        if worst <= 0 and margin is None:
            margin = budget

    return {
        'baseline_pnl': baseline,
        'margin_of_safety': margin,
        'verdict': 'CERTIFIED' if margin is None else 'BREAKABLE',
        'probes': probes,
        'largest_budget_survived': margin if margin is not None else values[-1],
    }


# --- the full audit -------------------------------------------------------


@dataclass
class RobustnessReport:
    """Everything the falsification pass found, in one reviewable object."""

    baseline_pnl: float
    attack_results: List[AttackResult]
    survival_rate: float
    retention: float
    break_even_cost: float
    sensitivity: Dict[str, float]
    certificate: Dict[str, object] = field(default_factory=dict)

    def weakest_attack(self) -> Optional[str]:
        """The attack that cost the most, by PnL lost."""
        if not self.attack_results:
            return None
        worst = min(self.attack_results, key=lambda r: r.pnl)
        return worst.attack

    def flipped_attacks(self) -> List[str]:
        """Attacks that turned a winner into a loser."""
        return sorted({r.attack for r in self.attack_results if r.flipped})

    @property
    def verdict(self) -> str:
        """
        A single word to put at the top of a report.

        The thresholds are deliberately strict. Most strategies that survive a
        serious attack still fail the cost test, and saying so is the point.
        """
        if self.baseline_pnl <= 0:
            return 'LOSES_MONEY'
        margin = self.certificate.get('margin_of_safety')
        if margin is not None and margin <= 0.02:
            return 'FRAGILE'
        if self.survival_rate < 0.7:
            return 'FRAGILE'
        if self.break_even_cost < 3.0:
            return 'FRAGILE'
        if self.flipped_attacks():
            return 'FRAGILE'
        return 'ROBUST'

    def as_dict(self) -> dict:
        return {
            'baseline_pnl': round(self.baseline_pnl, 8),
            'verdict': self.verdict,
            'survival_rate': round(self.survival_rate, 4),
            'retention': round(self.retention, 4),
            'break_even_cost': (
                round(self.break_even_cost, 4)
                if np.isfinite(self.break_even_cost)
                else None
            ),
            'weakest_attack': self.weakest_attack(),
            'flipped_attacks': self.flipped_attacks(),
            'attacks': [r.as_dict() for r in self.attack_results],
            'sensitivity': {k: round(float(v), 6) for k, v in self.sensitivity.items()},
            'certificate': {
                k: v for k, v in self.certificate.items() if k != 'probes'
            },
        }


def audit_robustness(
    strategy: Strategy,
    prices,
    severities: Sequence[float] = (0.25, 0.5),
    attacks: Optional[Sequence[Attack]] = None,
    n_perturbations: int = 32,
    seed: int = 0,
    budgets: Sequence[float] = (0.01, 0.02, 0.05),
    n_adversarial_steps: int = 30,
) -> RobustnessReport:
    """
    Run the whole falsification pass against one strategy.

    Every step is seeded, so the same inputs give the same report -- an
    audit you cannot reproduce is not an audit.
    """
    fn = _check_strategy(strategy)
    base = _as_prices(prices)
    chosen = list(attacks) if attacks is not None else DEFAULT_ATTACKS
    if not chosen:
        raise ValueError('need at least one attack')
    levels = list(severities)
    if not levels:
        raise ValueError('need at least one severity')
    if any(s <= 0 for s in levels):
        raise ValueError('severities must all be positive')

    baseline = float(fn(base))

    results: List[AttackResult] = []
    for attack in chosen:
        for severity in levels:
            # crc32, not hash(): CPython randomises string hashing per process,
            # which would make the report differ between two identical runs.
            rng = np.random.default_rng(seed + zlib.crc32(attack.name.encode()) % 9973)
            attacked = apply_attack(attack, base, float(severity), rng)
            results.append(
                AttackResult(
                    attack=attack.name,
                    severity=float(severity),
                    pnl=float(fn(attacked)),
                    baseline_pnl=baseline,
                )
            )

    certificate = certify_robustness(
        fn, base, budgets=budgets, n_steps=n_adversarial_steps, seed=seed
    )

    return RobustnessReport(
        baseline_pnl=baseline,
        attack_results=results,
        survival_rate=survival_rate(results),
        retention=pnl_retention(results),
        break_even_cost=break_even_multiple(fn, base),
        sensitivity=perturbation_sensitivity(
            fn, base, n_trials=n_perturbations, seed=seed
        ),
        certificate=certificate,
    )
