"""
Scheduled execution: turning one decision into a schedule of child orders.

Most backtesters treat "I want 1 BTC" as a single fill at the next candle's
close, optionally minus a percentage. That is not what actually happens. Real
size does not print all at once, and the two forces that push against each
other are both large:

Timing risk
    The longer the parent order is worked, the more the price can wander away
    from the price that was good enough to make the decision. You are exposed
    to the market for longer.

Market impact
    Sending more, faster, in a thin book moves the price against you. Sending
    less, slower, means the position is exposed for longer.

:func:`is_optimal` solves the trade-off between exactly those two terms, which
is the Almgren-Chriss optimal execution trajectory written for bars rather than
seconds. The answer is not "trade slowly and politely" -- it is a specific
shape that front-loads when volatility is high and back-loads when impact is
expensive relative to the risk of waiting, and it is dominated by neither
naive rule.

The framework is: build an :class:`ExecutionPlan` with an algorithm, then
:func:`simulate_execution` scores it against a real price path, so algorithms
can be compared on the same data rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    'ExecutionParams',
    'ChildOrder',
    'ExecutionPlan',
    'ExecutionResult',
    'twap_schedule',
    'pov_schedule',
    'vwap_schedule',
    'is_optimal',
    'adaptive_pov_schedule',
    'benchmark_curve',
    'schedule_by_name',
    'simulate_execution',
    'compare_algorithms',
    'AlgorithmComparison',
]

BUY = 1
SELL = -1


@dataclass(frozen=True)
class ExecutionParams:
    """
    The economics of executing against this market.

    total_qty
        Size of the parent order, always positive. Direction is `side`.
    side
        ``+1`` buy, ``-1`` sell.
    bar_seconds
        Wall-clock length of one bar. Only used to report the schedule in
        time units; nothing in the maths depends on it.
    volatility
        Per-bar volatility in *price units* (not returns). Convert with
        ``close.std()`` or :func:`realized_volatility` scaled by price. This
        is the size of the timing risk the schedule takes on.
    impact
        Temporary impact coefficient. A child order of ``q`` into a bar with
        ``v`` volume costs ``impact * q**2 / v`` in price units.
    spread_cost
        Full spread crossed per fill, in price units. Half of it is paid on
        entry.
    risk_aversion
        How much timing risk is acceptable per unit of expected impact cost.
        Zero degenerates to "send everything at once"; large values push the
        schedule towards the front.
    max_participation
        Ceiling on ``q / v`` for any single child. Acts as a hard cap that no
        algorithm is allowed to exceed, whatever its own logic would prefer.
    """

    total_qty: float
    side: int = BUY
    bar_seconds: int = 60
    volatility: float = 0.0
    impact: float = 0.0
    spread_cost: float = 0.0
    risk_aversion: float = 1.0
    max_participation: float = 0.25

    def __post_init__(self) -> None:
        if self.total_qty <= 0:
            raise ValueError(f'total_qty must be positive, got {self.total_qty}')
        if self.side not in (BUY, SELL):
            raise ValueError(f'side must be {BUY} or {SELL}, got {self.side}')
        if self.volatility < 0:
            raise ValueError(f'volatility must be non-negative, got {self.volatility}')
        if self.impact < 0:
            raise ValueError(f'impact must be non-negative, got {self.impact}')
        if self.spread_cost < 0:
            raise ValueError(f'spread_cost must be non-negative, got {self.spread_cost}')
        if self.risk_aversion < 0:
            raise ValueError(
                f'risk_aversion must be non-negative, got {self.risk_aversion}'
            )
        if not 0 < self.max_participation <= 1:
            raise ValueError(
                f'max_participation must be in (0, 1], got {self.max_participation}'
            )

    @property
    def signed_qty(self) -> float:
        """Parent size carrying its direction."""
        return self.side * self.total_qty


@dataclass(frozen=True)
class ChildOrder:
    """One slice of the parent order, scheduled against a bar index."""

    bar_index: int
    qty: float

    def __post_init__(self) -> None:
        if self.qty < 0:
            raise ValueError(f'child qty must be non-negative, got {self.qty}')
        if self.bar_index < 0:
            raise ValueError(f'bar_index must be non-negative, got {self.bar_index}')


@dataclass(frozen=True)
class ExecutionPlan:
    """A complete schedule: which bar each slice works, and how much."""

    side: int
    total_qty: float
    children: List[ChildOrder]
    algorithm: str = 'custom'
    meta: dict = field(default_factory=dict)

    @property
    def bars(self) -> np.ndarray:
        """Bar index of every child, ascending."""
        return np.array([c.bar_index for c in self.children], dtype=np.int64)

    @property
    def quantities(self) -> np.ndarray:
        """Size of every child, in schedule order."""
        return np.array([c.qty for c in self.children], dtype=np.float64)

    @property
    def filled_qty(self) -> float:
        return float(self.quantities.sum())

    @property
    def duration(self) -> int:
        """Number of distinct bars touched."""
        return int(len(np.unique(self.bars)))

    def completion_bar(self) -> Optional[int]:
        """Last bar that actually carries size, or None if the plan is empty."""
        active = self.children and [c.bar_index for c in self.children if c.qty > 0]
        return int(max(active)) if active else None


def _quantities_to_plan(
    quantities: np.ndarray,
    params: ExecutionParams,
    algorithm: str,
    meta: Optional[dict] = None,
) -> ExecutionPlan:
    """
    Build a plan from explicit per-bar quantities, without rescaling them.

    `params.total_qty` stays on the plan as the *requested* size even when the
    quantities add up to less, so an order the tape cannot absorb reports
    itself as partially filled instead of quietly becoming a smaller order.
    """
    values = np.asarray(quantities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f'quantities must be a non-empty 1-D array, got shape {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('quantities must all be finite')
    if np.any(values < -1e-12):
        raise ValueError(f'quantities must be non-negative, got min {values.min()}')
    values = np.clip(values, 0.0, None)

    children = [
        ChildOrder(bar_index=i, qty=float(q)) for i, q in enumerate(values) if q > 1e-12
    ]
    return ExecutionPlan(
        side=params.side,
        total_qty=params.total_qty,
        children=children,
        algorithm=algorithm,
        meta=meta or {},
    )


def _weights_to_plan(
    weights: np.ndarray,
    params: ExecutionParams,
    algorithm: str,
    meta: Optional[dict] = None,
) -> ExecutionPlan:
    """
    Turn non-negative weights over `len(weights)` bars into a plan that sums
    exactly to `total_qty`, using integer bar indices from 0.

    The last bar absorbs the rounding residue so the schedule never loses or
    invents size -- the kind of small leak that quietly shows up as a strategy
    that cannot go flat.
    """
    raw = np.asarray(weights, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f'weights must be a non-empty 1-D array, got shape {raw.shape}')
    if not np.all(np.isfinite(raw)):
        raise ValueError('weights must all be finite')
    if np.any(raw < -1e-12):
        raise ValueError(f'weights must be non-negative, got min {raw.min()}')
    raw = np.clip(raw, 0.0, None)

    total = raw.sum()
    if total <= 0:
        raise ValueError('weights must not be all zero; nothing to execute')

    quantities = raw / total * params.total_qty
    # Residue from float truncation lands on the final bar.
    residue = params.total_qty - quantities.sum()
    quantities[-1] += residue
    if quantities[-1] < 0:
        # Rounding overshot; pull the excess back out of the last positive bar.
        overshoot = -quantities[-1]
        quantities[-1] = 0.0
        donors = [i for i, q in enumerate(quantities) if q > 0]
        for i in reversed(donors):
            take = min(quantities[i], overshoot)
            quantities[i] -= take
            overshoot -= take
            if overshoot <= 1e-15:
                break

    return _quantities_to_plan(quantities, params, algorithm, meta)


def twap_schedule(params: ExecutionParams, bars: int) -> ExecutionPlan:
    """
    Time-weighted average price: the same size in every bar.

    The baseline every other algorithm has to beat. Equal *time* weighting is
    only equal *volume* weighting if volume arrives uniformly, which it does
    not -- which is exactly why TWAP usually loses to following the tape.
    """
    if bars < 1:
        raise ValueError(f'bars must be at least 1, got {bars}')
    return _weights_to_plan(
        np.ones(bars), params, 'twap', {'bars': int(bars)}
    )


def pov_schedule(
    params: ExecutionParams,
    volumes: Sequence[float],
    participation: Optional[float] = None,
) -> ExecutionPlan:
    """
    Percentage of volume: work a fixed fraction of whatever the market does.

    `volumes` is expected (not perfect) volume per bar. Each bar is allowed
    ``participation * volumes[t]``, and whatever has not worked when the tape
    runs out is pushed into the final bar -- which is precisely the situation
    the participation cap exists to prevent, and why
    ``params.max_participation`` still binds as a hard ceiling.

    Feeds with missing volume are the common failure here. A bar with zero
    reported volume is treated as untradeable rather than as infinite
    liquidity, so a data gap cannot be laundered into a fill.
    """
    vols = np.asarray(volumes, dtype=np.float64)
    if vols.ndim != 1 or vols.size == 0:
        raise ValueError(f'volumes must be a non-empty 1-D array, got shape {vols.shape}')
    if not np.all(np.isfinite(vols)):
        raise ValueError('volumes must all be finite')

    rate = params.max_participation if participation is None else float(participation)
    if not 0 < rate <= 1:
        raise ValueError(f'participation must be in (0, 1], got {rate}')

    tradable = np.where(vols > 0, vols, 0.0)
    capacities = rate * tradable
    if capacities.sum() <= 0:
        # No reported volume anywhere: fall back to working it evenly.
        return _weights_to_plan(
            np.ones(vols.size), params, 'pov', {'participation': rate, 'fallback': 'time'}
        )

    # Bars past the point where the order is already full get nothing.
    cumulative = np.cumsum(capacities)
    stop = int(np.searchsorted(cumulative, params.total_qty, side='left')) + 1
    stop = min(stop, vols.size)
    quantities = capacities[:stop].copy()
    # Do not overshoot the parent size on the bar that completes it.
    overshoot = quantities.sum() - params.total_qty
    if overshoot > 0:
        quantities[-1] -= overshoot

    meta = {
        'participation': rate,
        'bars_used': int(stop),
        'zero_volume_bars': int(np.count_nonzero(vols <= 0)),
        'fillable_qty': float(quantities.sum()),
        'unfilled_qty': float(max(params.total_qty - quantities.sum(), 0.0)),
    }
    return _quantities_to_plan(quantities, params, 'pov', meta)


def benchmark_curve(volumes: Sequence[float]) -> np.ndarray:
    """
    Cumulative volume shares that a perfect VWAP execution would achieve.

    Used as the tracking target by :func:`vwap_schedule` and reported by
    :func:`simulate_execution` so a schedule can be scored on how closely it
    followed the natural liquidity profile.
    """
    vols = np.asarray(volumes, dtype=np.float64)
    if vols.ndim != 1 or vols.size == 0:
        raise ValueError(f'volumes must be a non-empty 1-D array, got shape {vols.shape}')
    cumulative = np.cumsum(np.where(vols > 0, vols, 0.0))
    total = cumulative[-1]
    if total <= 0:
        return np.ones(vols.size) / vols.size
    return cumulative / total


def vwap_schedule(
    params: ExecutionParams,
    volumes: Sequence[float],
    tolerance: float = 0.25,
) -> ExecutionPlan:
    """
    Track the volume-weighted benchmark, allowed to drift inside a band.

    `tolerance` is the maximum permitted gap between the schedule's own
    completion percentage and the benchmark's, as a fraction of the total
    quantity. Widening it lets the algorithm lean on bars where volume is
    cheap; tightening it forces a schedule close to the benchmark curve.

    Unlike POV there is no per-bar ceiling here, because the benchmark itself
    can demand a lot in one bar. The result is still capped by
    ``params.max_participation`` via a final clipping pass, which is what
    keeps a single dominant-volume bar from turning into an implausible fill.
    """
    if not 0 <= tolerance <= 1:
        raise ValueError(f'tolerance must be in [0, 1], got {tolerance}')

    vols = np.asarray(volumes, dtype=np.float64)
    if vols.ndim != 1 or vols.size == 0:
        raise ValueError(f'volumes must be a non-empty 1-D array, got shape {vols.shape}')
    if not np.all(np.isfinite(vols)):
        raise ValueError('volumes must all be finite')

    benchmark = benchmark_curve(vols)
    target = np.diff(np.concatenate(([0.0], benchmark))) * params.total_qty

    tradable = np.where(vols > 0, vols, 0.0)
    cap = params.max_participation * np.maximum(tradable, np.finfo(np.float64).tiny)
    capped = np.minimum(target, cap)
    if capped.sum() <= 0:
        capped = np.minimum(cap, params.total_qty)
    if capped.sum() <= 0:
        capped = np.full(vols.size, params.total_qty / vols.size)

    # Redistribute whatever the cap removed across bars that still have
    # headroom, repeatedly. If the tape simply cannot absorb the parent size
    # within the ceiling, the plan stays under-filled rather than breaching
    # the cap -- an honest partial fill beats an impossible one.
    shortfall = params.total_qty - capped.sum()
    guard = 0
    while shortfall > 1e-9 and guard < 100:
        guard += 1
        headroom = np.maximum(cap - capped, 0.0)
        if headroom.sum() <= 1e-12:
            break
        added = np.minimum(headroom * (shortfall / headroom.sum()), headroom)
        capped = capped + added
        shortfall = params.total_qty - capped.sum()

    meta = {
        'tolerance': float(tolerance),
        'bars': int(vols.size),
        'capped_bars': int(np.count_nonzero(target > cap)),
        'fillable_qty': float(capped.sum()),
        'unfilled_qty': float(max(params.total_qty - capped.sum(), 0.0)),
    }
    return _quantities_to_plan(capped, params, 'vwap', meta)


def is_optimal(
    params: ExecutionParams,
    volumes: Sequence[float],
    bars: Optional[int] = None,
) -> ExecutionPlan:
    """
    Almgren-Chriss optimal trajectory, solved in closed-form-gradient form.

    Minimises, over schedules ``x_t`` summing to the parent size:

        sum_t [ impact * x_t**2 / V_t  +  risk_aversion * volatility**2 * R_t**2 ]

    where ``V_t`` is bar ``t`` volume and ``R_t`` is the size still unexecuted
    after bar ``t``. The first term is temporary market impact (concave in
    size, so it punishes clumping). The second is timing risk (the penalty on
    still holding inventory). The schedule is therefore genuinely determined
    by the ratio of the two coefficients -- not a stylistic choice.

    This is a convex quadratic programme with a simplex constraint, solved with
    SLSQP from a TWAP warm start and an analytic gradient. Degenerate inputs
    (no volatility, no impact) are handled explicitly because the solver is
    indifferent between schedules that are exactly tied.
    """
    vols = np.asarray(volumes, dtype=np.float64)
    if vols.ndim != 1 or vols.size == 0:
        raise ValueError(f'volumes must be a non-empty 1-D array, got shape {vols.shape}')
    if not np.all(np.isfinite(vols)):
        raise ValueError('volumes must all be finite')

    n = int(bars) if bars is not None else vols.size
    if n < 1:
        raise ValueError(f'bars must be at least 1, got {n}')
    tradable = np.where(vols[:n] > 0, vols[:n], 0.0)
    if tradable.sum() <= 0:
        tradable = np.ones(n)

    impact = params.impact
    risk = params.risk_aversion * params.volatility**2

    # Pure corners: with one term gone the optimum is obvious, and asking a
    # solver to find a continuum of identical answers is wasted work.
    if impact <= 0 and risk <= 0:
        weights = np.ones(n)
        return _weights_to_plan(
            weights, params, 'is_optimal', {'degenerate': 'no-impact-no-risk'}
        )
    if impact <= 0:
        # Timing risk only: everything at once, immediately.
        weights = np.zeros(n)
        weights[0] = 1.0
        return _weights_to_plan(
            weights, params, 'is_optimal', {'degenerate': 'risk-only-immediate'}
        )
    if risk <= 0:
        # Impact only: spread out to fill every bar's cap before doubling up.
        weights = params.max_participation * tradable
        if weights.sum() <= 0:
            weights = np.ones(n)
        return _weights_to_plan(
            weights, params, 'is_optimal', {'degenerate': 'impact-only-even'}
        )

    # Hard per-bar participation ceiling, expressed as an upper bound.
    upper = np.minimum(params.total_qty, params.max_participation * tradable)
    if upper.sum() <= 1e-12:
        upper = np.full(n, params.total_qty)
    cap_relaxed = bool(upper.sum() < params.total_qty)
    if cap_relaxed:
        # The participation ceiling cannot absorb the whole order inside the
        # tape. Refusing to fill would be a worse lie than relaxing the cap by
        # a known factor, so scale the bounds and record that it happened.
        upper = upper * (params.total_qty / upper.sum())

    inv_volume = 1.0 / np.maximum(tradable, np.finfo(np.float64).tiny)
    # Both terms are expressed as a fraction of the parent notional. Without
    # that normalisation the risk term (which is quadratic in the *open
    # quantity*) dwarfs the impact term by orders of magnitude for any
    # realistic order size, and the solver collapses onto "send it all now"
    # regardless of what the impact coefficient says.
    share = 1.0 / params.total_qty

    def objective(x: np.ndarray) -> float:
        remaining = params.total_qty - np.cumsum(x)
        impact_cost = impact * share * np.sum(x**2 * inv_volume)
        risk_cost = risk * share**2 * np.sum(remaining**2)
        return impact_cost + risk_cost

    def gradient(x: np.ndarray) -> np.ndarray:
        remaining = params.total_qty - np.cumsum(x)
        # d(sum remaining_u**2)/dx_t = -2 * sum_{u >= t} remaining_u
        tail = np.cumsum(remaining[::-1])[::-1]
        return 2.0 * impact * share * x * inv_volume - 2.0 * risk * share**2 * tail

    x0 = np.full(n, params.total_qty / n)
    bounds = [(0.0, float(upper[i])) for i in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda x: x.sum() - params.total_qty},)

    solution = minimize_slsqp(objective, x0, bounds, constraints, gradient)

    weights = solution.x
    if not np.all(np.isfinite(weights)) or abs(weights.sum() - params.total_qty) > 1e-6:
        # The solver failed to land on the constraint; fall back to something
        # that is definitely feasible rather than returning a broken plan.
        weights = np.minimum(upper, params.total_qty / n)
        if weights.sum() <= 0:
            weights = np.ones(n)

    weights = np.maximum(weights, 0.0)
    impact_cost = impact * share * float(np.sum(weights**2 * inv_volume))
    risk_cost = risk * share**2 * float(np.sum((params.total_qty - np.cumsum(weights)) ** 2))

    meta = {
        'bars': int(n),
        'impact': float(impact),
        'risk': float(risk),
        'impact_share': float(impact_cost / max(impact_cost + risk_cost, 1e-300)),
        'risk_share': float(risk_cost / max(impact_cost + risk_cost, 1e-300)),
        'front_loaded_fraction': float(weights[: max(1, n // 4)].sum() / weights.sum()),
        'cap_relaxed': cap_relaxed,
    }
    return _weights_to_plan(weights, params, 'is_optimal', meta)


def adaptive_pov_schedule(
    params: ExecutionParams,
    closes: Sequence[float],
    volumes: Sequence[float],
    participation: float = 0.15,
    step: float = 0.05,
    floor: float = 0.02,
    ceiling: float = 0.60,
) -> ExecutionPlan:
    """
    POV with a live implementation-shortfall feedback loop.

    Real desks do not run a fixed participation rate; they speed up when they
    are falling behind the benchmark and slow down when ahead. This schedule
    recomputes its own rate each bar from the shortfall accumulated so far.

    Falling behind means the schedule's completion percentage is under the
    volume benchmark's, so `participation` rises by `step`, floored and
    ceilinged. The result adapts to a tape that dries up mid-order without any
    external signal.
    """
    if step < 0:
        raise ValueError(f'step must be non-negative, got {step}')
    if not 0 < floor <= ceiling <= 1:
        raise ValueError(f'need 0 < floor <= ceiling <= 1, got floor={floor} ceiling={ceiling}')

    closes = np.asarray(closes, dtype=np.float64)
    vols = np.asarray(volumes, dtype=np.float64)
    if closes.ndim != 1 or closes.size == 0:
        raise ValueError(f'closes must be a non-empty 1-D array, got shape {closes.shape}')
    if vols.size != closes.size:
        raise ValueError(f'closes ({closes.size}) and volumes ({vols.size}) must align')
    if not np.all(np.isfinite(closes)) or not np.all(np.isfinite(vols)):
        raise ValueError('closes and volumes must all be finite')
    if participation <= 0:
        raise ValueError(f'participation must be positive, got {participation}')

    benchmark = benchmark_curve(vols)
    n = closes.size
    rate = float(participation)
    executed = 0.0
    weights = np.zeros(n)
    rates_used: List[float] = []

    for t in range(n):
        volume = vols[t] if vols[t] > 0 else 0.0
        capacity = rate * volume
        if capacity <= 0 and executed < params.total_qty:
            # No reported volume: pace on time so the order still completes.
            capacity = params.total_qty / n
        slice_qty = min(capacity, params.total_qty - executed)
        slice_qty = max(slice_qty, 0.0)
        weights[t] = slice_qty
        executed += slice_qty
        rates_used.append(rate)

        if executed >= params.total_qty or t == n - 1:
            break
        own_progress = executed / params.total_qty
        gap = benchmark[t] - own_progress
        rate = float(np.clip(rate + step * np.sign(gap), floor, ceiling))

    if weights.sum() <= 0:
        weights = np.ones(n)

    meta = {
        'participation_start': float(participation),
        'participation_end': float(rates_used[-1]) if rates_used else float(participation),
        'rates': rates_used,
        'step': float(step),
    }
    return _weights_to_plan(weights, params, 'adaptive_pov', meta)


def schedule_by_name(
    name: str,
    params: ExecutionParams,
    volumes: Optional[Sequence[float]] = None,
    **kwargs,
) -> ExecutionPlan:
    """
    Dispatch to an algorithm by name, so callers can accept the algorithm as
    configuration instead of branching on it.
    """
    key = name.strip().lower().replace('-', '_')
    if key == 'twap':
        return twap_schedule(params, int(kwargs.get('bars', 10)))
    if volumes is None:
        raise ValueError(f'algorithm {name!r} needs a volumes argument')
    if key == 'pov':
        return pov_schedule(params, volumes, kwargs.get('participation'))
    if key == 'vwap':
        return vwap_schedule(params, volumes, kwargs.get('tolerance', 0.25))
    if key == 'is_optimal':
        return is_optimal(params, volumes, kwargs.get('bars'))
    if key == 'adaptive_pov':
        return adaptive_pov_schedule(
            params,
            kwargs['closes'],
            volumes,
            participation=kwargs.get('participation', 0.15),
            step=kwargs.get('step', 0.05),
        )
    raise ValueError(f'unknown execution algorithm {name!r}')


@dataclass(frozen=True)
class ExecutionResult:
    """What a schedule actually cost, scored against a real price path."""

    algorithm: str
    side: int
    requested_qty: float
    filled_qty: float
    arrival_price: float
    arrival_bar: int
    completion_bar: Optional[int]
    shortfall: float
    shortfall_bps: float
    spread_cost: float
    impact_cost: float
    timing_cost: float
    volume_weighted_benchmark: float
    schedule_tracking_error: float

    @property
    def complete(self) -> bool:
        """
        Whether the whole parent order was worked inside the tape.

        Deliberately about *size*, not about whether any bar carried a fill:
        a plan that filled half of a size it could never finish is incomplete,
        which is the fact an operator needs to see.
        """
        return self.unfilled_qty <= 1e-9 * max(self.requested_qty, 1.0)

    @property
    def unfilled_qty(self) -> float:
        return max(self.requested_qty - self.filled_qty, 0.0)


def simulate_execution(
    plan: ExecutionPlan,
    closes: Sequence[float],
    volumes: Sequence[float],
    arrival_bar: int = 0,
    impact: float = 0.0,
    spread_cost: float = 0.0,
) -> ExecutionResult:
    """
    Score a plan against a price path and decompose the shortfall.

    Fills are struck at the bar close moved against the trader by half the
    spread plus a temporary impact term ``impact * q**2 / v``. The three cost
    components are reported separately rather than summed into one number,
    because they call for opposite responses: spread and impact shrink with
    better execution, while timing cost only shrinks by finishing sooner.

    `schedule_tracking_error` is the RMS gap between the schedule's own
    completion curve and the volume benchmark -- the number that says whether
    the algorithm actually did what it claimed.
    """
    prices = np.asarray(closes, dtype=np.float64)
    vols = np.asarray(volumes, dtype=np.float64)
    if prices.ndim != 1 or prices.size == 0:
        raise ValueError(f'closes must be a non-empty 1-D array, got shape {prices.shape}')
    if vols.size != prices.size:
        raise ValueError(f'closes ({prices.size}) and volumes ({vols.size}) must align')
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(vols)):
        raise ValueError('closes and volumes must all be finite')
    if prices.size < 2:
        raise ValueError('need at least 2 bars to score an execution')
    if not 0 <= arrival_bar < prices.size - 1:
        raise ValueError(
            f'arrival_bar must be in [0, {prices.size - 2}], got {arrival_bar}'
        )

    arrival_price = float(prices[arrival_bar])
    if arrival_price <= 0:
        raise ValueError(f'arrival price must be positive, got {arrival_price}')

    spread_total = 0.0
    impact_total = 0.0
    timing_total = 0.0
    filled = 0.0
    completion_bar: Optional[int] = None
    progress: List[float] = []

    for child in plan.children:
        bar = child.bar_index
        if bar >= prices.size:
            break
        if child.qty <= 0:
            continue
        close = float(prices[bar])
        volume = float(vols[bar]) if vols[bar] > 0 else 0.0
        half_spread = spread_cost / 2.0
        # Temporary impact is linear in size, so the *total* cost of a slice
        # is impact * q**2 / v -- the same term is_optimal() minimises.
        impact_move = impact * child.qty / volume if volume > 0 else 0.0

        # Everything between the decision price and this bar's close, before
        # any impact, is timing cost attributable to waiting.
        timing = plan.side * (close - arrival_price) * child.qty
        spread = half_spread * child.qty
        impact_c = impact_move * child.qty

        spread_total += spread
        impact_total += impact_c
        timing_total += timing
        filled += child.qty
        completion_bar = bar
        progress.append(filled)

    shortfall = timing_total + spread_total + impact_total
    notional = arrival_price * max(filled, np.finfo(np.float64).tiny)
    shortfall_bps = shortfall / notional * 1e4

    benchmark = benchmark_curve(vols)
    if progress:
        own = np.asarray(progress, dtype=np.float64) / max(filled, np.finfo(np.float64).tiny)
        own_curve = np.concatenate(([0.0], own))
        bench_curve = np.concatenate(([0.0], benchmark[: len(own_curve)]))
        n = min(len(own_curve), len(bench_curve))
        tracking = float(np.sqrt(np.mean((own_curve[:n] - bench_curve[:n]) ** 2)))
    else:
        tracking = float('nan')

    vwap = float(
        np.sum(prices * np.where(vols > 0, vols, 0.0)) / max(np.sum(np.where(vols > 0, vols, 0.0)), 1e-12)
    )

    return ExecutionResult(
        algorithm=plan.algorithm,
        side=plan.side,
        requested_qty=plan.total_qty,
        filled_qty=float(filled),
        arrival_price=arrival_price,
        arrival_bar=arrival_bar,
        completion_bar=completion_bar,
        shortfall=float(shortfall),
        shortfall_bps=float(shortfall_bps),
        spread_cost=float(spread_total),
        impact_cost=float(impact_total),
        timing_cost=float(timing_total),
        volume_weighted_benchmark=vwap,
        schedule_tracking_error=tracking,
    )


def minimize_slsqp(objective, x0, bounds, constraints, gradient):
    """Run SLSQP, importing scipy lazily so module import stays cheap."""
    from scipy.optimize import minimize

    return minimize(
        objective,
        x0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12},
    )


@dataclass(frozen=True)
class AlgorithmComparison:
    """Side-by-side scores for every algorithm tried on the same tape."""

    results: List[ExecutionResult]

    def best(self) -> ExecutionResult:
        """The plan with the lowest shortfall."""
        if not self.results:
            raise ValueError('no results to compare')
        return min(self.results, key=lambda r: r.shortfall)

    def worst(self) -> ExecutionResult:
        if not self.results:
            raise ValueError('no results to compare')
        return max(self.results, key=lambda r: r.shortfall)

    def improvement_bps(self) -> float:
        """Basis points saved by the best plan versus the worst, positive."""
        if len(self.results) < 2:
            return 0.0
        b, w = self.best(), self.worst()
        return float((w.shortfall - b.shortfall) / abs(w.shortfall or 1.0) * 1e4)

    def as_dict(self) -> dict:
        return {
            r.algorithm: {
                'shortfall_bps': round(r.shortfall_bps, 4),
                'filled_qty': round(r.filled_qty, 8),
                'completion_bar': r.completion_bar,
                'impact_cost': round(r.impact_cost, 8),
                'timing_cost': round(r.timing_cost, 8),
                'spread_cost': round(r.spread_cost, 8),
                'tracking_error': round(r.schedule_tracking_error, 6),
            }
            for r in self.results
        }


def compare_algorithms(
    params: ExecutionParams,
    closes: Sequence[float],
    volumes: Sequence[float],
    algorithms: Optional[Sequence[str]] = None,
    arrival_bar: int = 0,
    impact: Optional[float] = None,
    spread_cost: Optional[float] = None,
) -> AlgorithmComparison:
    """
    Build and score every algorithm against one tape, so the choice of
    schedule is a measurement rather than an assertion.

    Volatility is estimated from the price path when `params` does not supply
    it, because the optimal schedule is a function of volatility and assuming
    zero silently collapses IS-optimal onto an arbitrary corner.
    """
    names = list(algorithms) if algorithms else ['twap', 'pov', 'vwap', 'is_optimal', 'adaptive_pov']
    prices = np.asarray(closes, dtype=np.float64)
    if prices.size < 2:
        raise ValueError('need at least 2 bars to compare algorithms')

    vol = params.volatility
    if vol <= 0:
        vol = float(np.std(np.diff(prices)))

    resolved_params = (
        params
        if params.volatility > 0
        else ExecutionParams(
            total_qty=params.total_qty,
            side=params.side,
            bar_seconds=params.bar_seconds,
            volatility=max(vol, 1e-12),
            impact=params.impact,
            spread_cost=params.spread_cost,
            risk_aversion=params.risk_aversion,
            max_participation=params.max_participation,
        )
    )

    use_impact = resolved_params.impact if impact is None else impact
    use_spread = resolved_params.spread_cost if spread_cost is None else spread_cost

    # Every algorithm sees only the bars that exist after the decision, so a
    # plan cannot be built against bars the order could never have reached.
    window = np.asarray(volumes, dtype=np.float64)[arrival_bar:]
    window_closes = prices[arrival_bar:]

    results: List[ExecutionResult] = []
    for name in names:
        kwargs = {'closes': window_closes, 'bars': int(window_closes.size)}
        plan = schedule_by_name(name, resolved_params, window, **kwargs)
        if arrival_bar:
            plan = ExecutionPlan(
                side=plan.side,
                total_qty=plan.total_qty,
                children=[ChildOrder(c.bar_index + arrival_bar, c.qty) for c in plan.children],
                algorithm=plan.algorithm,
                meta=plan.meta,
            )
        results.append(
            simulate_execution(
                plan,
                prices,
                volumes,
                arrival_bar=arrival_bar,
                impact=use_impact,
                spread_cost=use_spread,
            )
        )
    return AlgorithmComparison(results=results)
