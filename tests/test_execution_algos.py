import numpy as np
import pytest

from algorithex.execution.algos import (
    BUY,
    SELL,
    AlgorithmComparison,
    ChildOrder,
    ExecutionParams,
    ExecutionPlan,
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


def tape(seed=11, n=40, start=100.0, drift=0.0):
    """A price path with a U-shaped volume profile."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.4, n)
    closes = start + np.cumsum(steps)
    bar = np.arange(n)
    volumes = 300.0 + 240.0 * np.sin(np.pi * bar / max(n - 1, 1))
    return closes, volumes


def params(**kwargs):
    base = dict(total_qty=100.0, volatility=0.4, impact=50.0, spread_cost=0.2, risk_aversion=1.0)
    base.update(kwargs)
    return ExecutionParams(**base)


def documented_objective(plan, p, volumes):
    """The objective is_optimal() documents, recomputed from the plan itself."""
    bars, quantities = plan.bars, plan.quantities
    share = 1.0 / p.total_qty
    impact_cost = p.impact * share * np.sum(quantities**2 / np.asarray(volumes)[bars])
    risk_cost = p.risk_aversion * p.volatility**2 * share**2 * np.sum(
        (p.total_qty - np.cumsum(quantities)) ** 2
    )
    return impact_cost + risk_cost


def front_share(plan):
    active = plan.quantities[plan.quantities > 0]
    return float(active[: max(1, len(active) // 4)].sum() / active.sum())


# --- params ---------------------------------------------------------------


def test_params_reject_impossible_settings():
    with pytest.raises(ValueError):
        ExecutionParams(total_qty=0)
    with pytest.raises(ValueError):
        ExecutionParams(total_qty=1.0, side=0)
    with pytest.raises(ValueError):
        ExecutionParams(total_qty=1.0, impact=-1.0)
    with pytest.raises(ValueError):
        ExecutionParams(total_qty=1.0, max_participation=0.0)
    with pytest.raises(ValueError):
        ExecutionParams(total_qty=1.0, max_participation=1.5)


def test_signed_qty_carries_direction():
    assert params().signed_qty == 100.0
    assert params(side=SELL).signed_qty == -100.0


# --- schedules ------------------------------------------------------------


def test_twap_splits_evenly_and_conserves_size():
    p = params()
    plan = twap_schedule(p, 10)

    assert plan.filled_qty == pytest.approx(100.0)
    assert len(plan.children) == 10
    quantities = plan.quantities
    assert quantities.std() < 1e-9
    assert plan.completion_bar() == 9


def test_every_plan_conserves_the_parent_size_exactly():
    """Float truncation must never leave a residue the strategy cannot close."""
    closes, volumes = tape()
    plans = [
        twap_schedule(params(total_qty=7.0), 13),
        pov_schedule(params(total_qty=7.0), volumes),
        vwap_schedule(params(total_qty=7.0), volumes),
        is_optimal(params(total_qty=7.0), volumes),
        adaptive_pov_schedule(params(total_qty=7.0), closes, volumes),
    ]
    for plan in plans:
        assert plan.filled_qty == pytest.approx(7.0, abs=1e-9), plan.algorithm


def test_an_unfillable_participation_order_reports_itself_incomplete():
    """Refusing to fill is honest; silently shrinking the order is not."""
    _, volumes = tape()
    plan = pov_schedule(params(total_qty=10_000.0, max_participation=0.10), volumes)

    assert plan.filled_qty < 10_000.0
    assert plan.total_qty == 10_000.0
    assert plan.meta['unfilled_qty'] == pytest.approx(10_000.0 - plan.filled_qty)


def test_pov_never_takes_more_than_the_configured_share_of_a_bar():
    _, volumes = tape()
    p = params(total_qty=10_000.0, max_participation=0.10)
    plan = pov_schedule(p, volumes)

    for child in plan.children:
        assert child.qty <= 0.10 * volumes[child.bar_index] + 1e-6


def test_pov_treats_missing_volume_as_untradeable_not_as_infinite_liquidity():
    volumes = np.array([0.0, 0.0, 500.0, 500.0, 0.0])
    plan = pov_schedule(params(total_qty=100.0), volumes, participation=0.20)

    assert all(child.bar_index != 0 for child in plan.children)
    assert all(child.bar_index != 1 for child in plan.children)
    assert plan.filled_qty == pytest.approx(100.0)


def test_pov_falls_back_to_timing_when_no_volume_is_reported():
    plan = pov_schedule(params(total_qty=100.0), np.zeros(5))
    assert plan.filled_qty == pytest.approx(100.0)
    assert plan.meta['fallback'] == 'time'


def test_vwap_follows_the_volume_profile_not_the_clock():
    volumes = np.array([10.0, 10.0, 10.0, 10.0, 1000.0])
    plan = vwap_schedule(params(total_qty=1000.0), volumes)

    # The benchmark says ~98% of the order works in the final bar; a
    # time-weighted schedule would have put 20% there.
    assert plan.quantities[-1] / plan.quantities.sum() > 0.5


def test_vwap_respects_the_participation_ceiling():
    volumes = np.array([10.0, 10.0, 10.0, 10.0, 100_000.0])
    p = params(total_qty=100_000.0, max_participation=0.05)
    plan = vwap_schedule(p, volumes)

    for child in plan.children:
        assert child.qty <= 0.05 * volumes[child.bar_index] + 1e-6


def test_vwap_tolerance_is_validated():
    with pytest.raises(ValueError):
        vwap_schedule(params(), [1.0, 2.0], tolerance=1.5)


def test_is_optimal_front_loads_when_timing_risk_dominates():
    """The shape must be a function of the coefficient ratio, not a style."""
    _, volumes = tape()
    patient = is_optimal(params(impact=500.0, risk_aversion=0.01), volumes)
    urgent = is_optimal(params(impact=0.05, risk_aversion=100.0), volumes)

    assert front_share(urgent) > front_share(patient)


def test_is_optimal_back_loads_when_impact_is_expensive():
    _, volumes = tape()
    cheap = is_optimal(params(impact=0.05, risk_aversion=100.0), volumes)
    expensive = is_optimal(params(impact=500.0, risk_aversion=0.01), volumes)

    assert front_share(cheap) > front_share(expensive)


def test_is_optimal_is_not_simply_either_corner():
    """The whole point: a real solution, neither TWAP nor send-it-all."""
    _, volumes = tape()
    plan = is_optimal(params(), volumes)

    assert plan.quantities.min() > 0
    assert plan.quantities.std() > 0
    assert 0.05 < front_share(plan) < 0.99


def test_is_optimal_beats_both_naive_corners_on_the_objective_it_solves():
    _, volumes = tape()
    p = params()

    best = documented_objective(is_optimal(p, volumes), p, volumes)
    twap = documented_objective(twap_schedule(p, volumes.size), p, volumes)
    all_now = documented_objective(
        ExecutionPlan(side=BUY, total_qty=p.total_qty, children=[ChildOrder(0, p.total_qty)]),
        p,
        volumes,
    )

    assert best < twap
    assert best < all_now


def test_is_optimal_reports_which_term_is_driving_the_schedule():
    _, volumes = tape()
    greedy = is_optimal(params(impact=0.05, risk_aversion=100.0), volumes)
    cautious = is_optimal(params(impact=500.0, risk_aversion=0.01), volumes)

    assert greedy.meta['risk_share'] > cautious.meta['risk_share']


def test_is_optimal_handles_the_degenerate_corners_without_the_solver():
    volumes = np.array([100.0, 100.0, 100.0])

    no_impact = is_optimal(params(impact=0.0, risk_aversion=10.0), volumes)
    assert no_impact.completion_bar() == 0

    no_risk = is_optimal(params(impact=1.0, risk_aversion=0.0, volatility=0.0), volumes)
    assert no_risk.duration == 3


def test_is_optimal_relaxes_the_cap_when_the_tape_cannot_absorb_the_order():
    """A hard refusal to fill is a worse lie than a recorded cap relaxation."""
    thin = np.full(10, 1.0)
    plan = is_optimal(params(total_qty=10_000.0, max_participation=0.01), thin)
    assert plan.filled_qty == pytest.approx(10_000.0)


def test_adaptive_pov_speeds_up_when_it_falls_behind_the_benchmark():
    # Volume accelerates, so the benchmark pulls ahead of a passive schedule
    # and the controller has to raise its own participation rate to catch up.
    volumes = np.array([1.0] * 8 + [100.0] * 2)
    closes = np.full(10, 100.0)
    plan = adaptive_pov_schedule(
        params(total_qty=1000.0), closes, volumes, participation=0.10, step=0.05
    )

    assert plan.meta['participation_end'] > plan.meta['participation_start']


def test_adaptive_pov_slows_down_when_it_is_ahead_of_the_benchmark():
    # Even volume, but a participation rate above the benchmark's share, so the
    # schedule is consistently ahead and the controller should ease off.
    volumes = np.full(10, 900.0)
    closes = np.full(10, 100.0)
    plan = adaptive_pov_schedule(
        params(total_qty=3000.0), closes, volumes, participation=0.5, step=0.05
    )

    assert plan.meta['participation_end'] < plan.meta['participation_start']


def test_adaptive_pov_respects_its_own_ceiling():
    volumes = np.array([1.0] * 6 + [9000.0] * 4)
    closes = np.full(10, 100.0)
    plan = adaptive_pov_schedule(
        params(total_qty=5000.0), closes, volumes, participation=0.05,
        step=0.5, floor=0.01, ceiling=0.30,
    )
    assert max(plan.meta['rates']) <= 0.30 + 1e-12


def test_adaptive_pov_validates_its_band():
    with pytest.raises(ValueError):
        adaptive_pov_schedule(params(), [1.0], [1.0], floor=0.5, ceiling=0.2)
    with pytest.raises(ValueError):
        adaptive_pov_schedule(params(), [1.0], [1.0], step=-1.0)


# --- simulation -----------------------------------------------------------


def test_simulate_decomposes_shortfall_into_the_three_pieces():
    closes, volumes = tape()
    p = params(total_qty=100.0)
    plan = twap_schedule(p, 10)
    result = simulate_execution(plan, closes, volumes, impact=0.5, spread_cost=0.2)

    assert result.shortfall == pytest.approx(
        result.timing_cost + result.spread_cost + result.impact_cost
    )
    assert result.spread_cost > 0
    assert result.impact_cost > 0
    assert result.filled_qty == pytest.approx(100.0)


def test_shortfall_is_measured_against_the_arrival_price():
    """A flat tape with pure impact must cost exactly the impact term."""
    closes = np.full(10, 100.0)
    volumes = np.full(10, 1000.0)
    plan = twap_schedule(params(total_qty=100.0, impact=1.0), 10)
    result = simulate_execution(plan, closes, volumes, impact=1.0, spread_cost=0.0)

    # Temporary impact is linear in size, so ten 10-lot slices each cost
    # impact * q**2 / v -- not impact * q**3 / v.
    expected = 10 * (1.0 * 10.0**2 / 1000.0)
    assert result.timing_cost == pytest.approx(0.0)
    assert result.shortfall == pytest.approx(expected)


def test_a_buy_into_a_rising_tape_pays_timing_cost():
    closes = np.arange(20, dtype=float) + 100.0
    volumes = np.full(20, 1000.0)
    plan = twap_schedule(params(total_qty=100.0), 5)
    result = simulate_execution(plan, closes, volumes)

    assert result.timing_cost > 0


def test_a_sell_into_a_rising_tape_earns_timing_cost():
    closes = np.arange(20, dtype=float) + 100.0
    volumes = np.full(20, 1000.0)
    plan = twap_schedule(params(total_qty=100.0, side=SELL), 5)
    result = simulate_execution(plan, closes, volumes)

    assert result.timing_cost < 0


def test_working_faster_reduces_timing_cost_on_a_trending_tape():
    closes = np.arange(60, dtype=float) + 100.0
    volumes = np.full(60, 10_000.0)
    p = params(total_qty=1000.0)

    slow = simulate_execution(twap_schedule(p, 40), closes, volumes)
    fast = simulate_execution(twap_schedule(p, 3), closes, volumes)

    assert fast.timing_cost < slow.timing_cost


def test_simulate_rejects_misaligned_and_short_input():
    with pytest.raises(ValueError):
        simulate_execution(twap_schedule(params(), 3), [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        simulate_execution(twap_schedule(params(), 1), [1.0], [1.0])
    with pytest.raises(ValueError):
        simulate_execution(twap_schedule(params(), 3), [1.0, 2.0, 3.0], [1.0, 1.0, 1.0],
                           arrival_bar=2)


def test_a_plan_that_runs_off_the_end_reports_itself_incomplete():
    closes, volumes = tape(n=6)
    plan = ExecutionPlan(
        side=BUY, total_qty=100.0, children=[ChildOrder(0, 50.0), ChildOrder(99, 50.0)]
    )
    result = simulate_execution(plan, closes, volumes)

    assert not result.complete
    assert result.filled_qty == pytest.approx(50.0)
    assert result.unfilled_qty == pytest.approx(50.0)


def test_tracking_error_rewards_actually_following_the_volume_profile():
    closes = np.full(30, 100.0)
    volumes = np.array([10.0] * 15 + [1000.0] * 15)
    p = params(total_qty=2000.0)

    good = simulate_execution(vwap_schedule(p, volumes), closes, volumes)
    bad = simulate_execution(twap_schedule(p, 30), closes, volumes)

    assert good.schedule_tracking_error < bad.schedule_tracking_error


def test_benchmark_curve_ends_at_one_and_handles_dead_tape():
    curve = benchmark_curve([1.0, 1.0, 2.0])
    assert curve[-1] == pytest.approx(1.0)
    assert curve[0] == pytest.approx(0.25)

    dead = benchmark_curve([0.0, 0.0, 0.0])
    assert np.allclose(dead, 1 / 3)


# --- dispatch and comparison ---------------------------------------------


def test_schedule_by_name_dispatches_and_rejects_typos():
    closes, volumes = tape()
    p = params()
    assert schedule_by_name('TWAP', p, bars=5).algorithm == 'twap'
    assert schedule_by_name('is-optimal', p, volumes).algorithm == 'is_optimal'
    assert schedule_by_name('adaptive_pov', p, volumes, closes=closes).algorithm == 'adaptive_pov'

    with pytest.raises(ValueError, match='unknown execution algorithm'):
        schedule_by_name('iceberg', p, volumes)
    with pytest.raises(ValueError, match='needs a volumes argument'):
        schedule_by_name('vwap', p)


def test_compare_algorithms_scores_every_plan_on_the_same_tape():
    closes, volumes = tape(n=30)
    comparison = compare_algorithms(params(total_qty=200.0), closes, volumes)

    assert isinstance(comparison, AlgorithmComparison)
    assert len(comparison.results) == 5
    for result in comparison.results:
        assert result.filled_qty == pytest.approx(200.0, rel=0.02)
    assert comparison.best().shortfall <= comparison.worst().shortfall
    assert comparison.improvement_bps() >= 0
    assert set(comparison.as_dict()) == {
        'twap', 'pov', 'vwap', 'is_optimal', 'adaptive_pov'
    }


def test_comparison_honours_a_late_arrival_bar():
    """Algorithms must not plan against bars that existed before the decision."""
    closes, volumes = tape(n=30)
    comparison = compare_algorithms(
        params(total_qty=200.0), closes, volumes, arrival_bar=10
    )
    for result in comparison.results:
        assert result.completion_bar >= 10
        assert result.filled_qty == pytest.approx(200.0, rel=0.05)


def test_comparison_of_a_single_algorithm_has_no_improvement():
    closes, volumes = tape(n=20)
    comparison = compare_algorithms(params(total_qty=50.0), closes, volumes, ['twap'])
    assert comparison.improvement_bps() == 0.0


def test_comparison_estimates_volatility_when_the_caller_omits_it():
    """Zero volatility silently collapses IS-optimal onto an arbitrary corner."""
    closes, volumes = tape(n=25)
    comparison = compare_algorithms(
        ExecutionParams(total_qty=100.0, impact=0.5, spread_cost=0.2), closes, volumes
    )
    optimal = next(r for r in comparison.results if r.algorithm == 'is_optimal')
    assert optimal.filled_qty == pytest.approx(100.0)


def test_empty_comparison_cannot_be_ranked():
    with pytest.raises(ValueError):
        AlgorithmComparison(results=[]).best()
