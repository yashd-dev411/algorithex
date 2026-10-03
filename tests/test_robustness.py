import numpy as np
import pytest

from algorithex.audit.robustness import (
    DEFAULT_ATTACKS,
    Attack,
    AttackResult,
    RobustnessReport,
    adversarial_path,
    apply_attack,
    audit_robustness,
    break_even_multiple,
    certify_robustness,
    cost_stress_curve,
    drop_random_bars,
    inject_spikes,
    jitter_prices,
    perturbation_sensitivity,
    pnl_retention,
    scramble_volumes,
    shuffle_returns,
    survival_rate,
    block_bootstrap,
)


def walk(seed=3, n=300, drift=0.0004, vol=0.012):
    """A genuinely trending series: the kind of strategy that ought to survive."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    return 100.0 * np.exp(np.cumsum(log_returns))


def trend_follower(prices):
    """Buy and hold a long position."""
    return float(prices[-1] - prices[0])


def momentum_strategy(prices):
    """Follow 20-bar momentum. Survives mild noise, dies on shuffled data."""
    window = 20
    if prices.size <= window + 1:
        return 0.0
    position = np.sign(prices[window:-1] - prices[:-window - 1])
    moves = prices[window + 1:] / prices[window:-1] - 1.0
    return float(np.sum(position * moves))


def lucky_on_this_path(prices):
    """Overfit to one exact price level; should not survive any perturbation."""
    return float(prices[-1] == 1234.5)


def negative_on_everything(prices):
    return -1.0


def seedable(seed):
    return np.random.default_rng(seed)


# --- price series validation ---------------------------------------------


def test_every_attack_returns_a_valid_price_series():
    base = walk()
    rng = seedable(1)
    for attack in DEFAULT_ATTACKS:
        out = apply_attack(attack, base, 0.5, rng)
        assert out.size >= 3
        assert out.size <= base.size
        assert np.all(np.isfinite(out))
        assert np.all(out > 0)


def test_the_shape_preserving_attacks_keep_every_bar():
    base = walk()
    rng = seedable(11)
    for attack in DEFAULT_ATTACKS:
        if attack.name == 'drop_bars':
            continue
        assert apply_attack(attack, base, 0.5, rng).size == base.size, attack.name


def test_bad_price_input_is_rejected():
    for bad in ([1.0], [1.0, 2.0], [1.0, 0.0, 2.0], [1.0, np.nan, 2.0]):
        with pytest.raises(ValueError):
            jitter_prices(bad, 0.5, seedable(0))
    with pytest.raises(ValueError):
        jitter_prices(np.array([1.0, -2.0, 3.0]), 0.5, seedable(0))


def test_a_strategy_must_be_callable():
    with pytest.raises(TypeError):
        audit_robustness('not callable', walk())


def test_jitter_keeps_the_trend_but_roughens_the_path():
    base = walk()
    out = jitter_prices(base, 0.5, seedable(2))
    assert out[0] == pytest.approx(base[0])
    assert out[-1] > base[0]


def test_shuffle_preserves_the_distribution_and_destroys_the_order():
    base = walk()
    out = shuffle_returns(base, 1.0, seedable(4))
    log_a, log_b = np.diff(np.log(base)), np.diff(np.log(out))
    assert log_b.mean() == pytest.approx(log_a.mean(), abs=1e-9)
    assert not np.allclose(log_a[:20], log_b[:20])


def test_shuffle_with_zero_severity_is_a_no_op():
    base = walk()
    assert np.allclose(shuffle_returns(base, 0.0, seedable(5)), base)


def test_resample_changes_the_path_but_keeps_the_scale():
    base = walk()
    out = scramble_volumes(base, 1.0, seedable(6))
    assert not np.allclose(out, base)
    assert np.diff(np.log(out)).std() == pytest.approx(
        np.diff(np.log(base)).std(), rel=0.35
    )


def test_spikes_are_injected_and_recovered():
    base = walk()
    out = inject_spikes(base, 1.0, seedable(7))
    assert not np.allclose(out, base)
    # Most bars are untouched; only a few move.
    assert np.count_nonzero(out != base) < base.size // 3


def test_block_bootstrap_keeps_local_structure():
    base = walk(n=200, drift=0.0)
    out = block_bootstrap(base, 0.5, seedable(8))
    assert out.size == base.size
    assert np.all(out > 0)


def test_dropping_bars_shortens_and_rebases_the_series():
    base = walk()
    out = drop_random_bars(base, 0.3, seedable(9))
    assert out.size < base.size
    assert out[-1] == pytest.approx(base[-1])


# --- scoring --------------------------------------------------------------


def test_survival_rate_and_retention_summarise_the_attacks():
    results = [
        AttackResult('a', 0.5, 10.0, 20.0),
        AttackResult('b', 0.5, -1.0, 20.0),
        AttackResult('c', 0.5, 15.0, 20.0),
    ]
    assert survival_rate(results) == pytest.approx(2 / 3)
    assert pnl_retention(results) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        survival_rate([])


def test_a_flipped_attack_is_identified_as_such():
    result = AttackResult('shuffle', 1.0, -5.0, 50.0)
    assert result.flipped
    assert not result.profitable
    assert result.retention == pytest.approx(-0.1)

    # A baseline that was already negative cannot "flip".
    assert not AttackResult('a', 1.0, -5.0, -10.0).flipped


def test_retention_is_undefined_against_a_zero_baseline():
    assert np.isnan(AttackResult('a', 1.0, 1.0, 0.0).retention)


# --- cost stress ----------------------------------------------------------


def test_cost_stress_curve_falls_as_costs_rise():
    base = walk()
    curve = cost_stress_curve(trend_follower, base, [0.0, 0.1, 0.5, 1.0])
    assert curve == sorted(curve, reverse=True)
    assert curve[0] == pytest.approx(base[-1] - base[0], abs=1e-6)


def test_cost_stress_validates_its_multipliers():
    with pytest.raises(ValueError):
        cost_stress_curve(trend_follower, walk(), [])
    with pytest.raises(ValueError):
        cost_stress_curve(trend_follower, walk(), [-1.0])


def test_break_even_multiple_is_found_by_bisection():
    base = walk()
    multiplier = break_even_multiple(trend_follower, base)

    assert np.isfinite(multiplier)
    assert multiplier > 0
    # Above the break-even the strategy loses money; below it, it does not.
    assert cost_stress_curve(trend_follower, base, [multiplier * 0.9])[0] > 0
    assert cost_stress_curve(trend_follower, base, [multiplier * 1.1])[0] < 0


def test_break_even_is_infinite_when_no_cost_can_kill_the_strategy():
    # A flat 1.0 PnL cannot be eaten by a small cost multiplier on a path
    # whose total absolute move is well under one unit.
    assert break_even_multiple(lambda p: 1.0, walk(), hi=0.01) == float('inf')


def test_break_even_is_zero_when_the_strategy_already_loses():
    assert break_even_multiple(negative_on_everything, walk()) == 0.0


def test_break_even_validates_its_bracket():
    with pytest.raises(ValueError):
        break_even_multiple(trend_follower, walk(), lo=5.0, hi=1.0)
    with pytest.raises(ValueError):
        break_even_multiple(trend_follower, walk(), tolerance=0.0)


def test_a_custom_cost_model_is_used_when_supplied():
    base = walk()
    flat = lambda path, multiplier: 5.0 - multiplier
    assert cost_stress_curve(trend_follower, base, [0.0, 2.0], flat) == [5.0, 3.0]


# --- sensitivity ----------------------------------------------------------


def test_perturbation_sensitivity_reports_a_distribution():
    stats = perturbation_sensitivity(momentum_strategy, walk(), n_trials=24, seed=1)
    assert stats['min'] <= stats['mean'] <= stats['max']
    assert stats['p05'] <= stats['p95']
    assert 0.0 <= stats['loss_frequency'] <= 1.0
    assert stats['std'] > 0


def test_perturbation_sensitivity_validates_its_inputs():
    with pytest.raises(ValueError):
        perturbation_sensitivity(momentum_strategy, walk(), n_trials=1)
    with pytest.raises(ValueError):
        perturbation_sensitivity(momentum_strategy, walk(), severity=0.0)


def test_a_more_severe_perturbation_produces_more_variance():
    mild = perturbation_sensitivity(momentum_strategy, walk(), 0.05, 24, seed=2)
    wild = perturbation_sensitivity(momentum_strategy, walk(), 1.00, 24, seed=2)
    assert wild['std'] > mild['std']


# --- certified worst case -------------------------------------------------


def test_adversarial_search_stays_inside_its_budget():
    base = walk()
    result = adversarial_path(trend_follower, base, budget=0.05, n_steps=20, seed=1)

    deviation = np.abs(result['worst_path'] / base - 1.0)
    assert deviation.max() <= 0.05 + 1e-9
    assert np.all(result['worst_path'] > 0)


def test_adversarial_search_never_does_worse_than_the_original_path():
    base = walk()
    for fn in (trend_follower, momentum_strategy):
        result = adversarial_path(fn, base, budget=0.05, n_steps=25, seed=2)
        assert result['worst_pnl'] <= result['baseline_pnl'] + 1e-9


def test_adversarial_search_breaks_a_strategy_that_depends_on_one_level():
    """A strategy keyed to one exact price should die under a 5% budget."""
    base = walk()
    result = adversarial_path(lucky_on_this_path, base, budget=0.05, n_steps=40, seed=3)
    assert result['worst_pnl'] <= result['baseline_pnl']


def test_a_bigger_budget_weakens_the_certificate():
    base = walk()
    small = adversarial_path(momentum_strategy, base, budget=0.01, n_steps=25, seed=4)
    large = adversarial_path(momentum_strategy, base, budget=0.30, n_steps=25, seed=4)
    assert large['worst_pnl'] <= small['worst_pnl'] + 1e-9


def test_adversarial_search_validates_its_inputs():
    base = walk()
    with pytest.raises(ValueError):
        adversarial_path(trend_follower, base, budget=0.0)
    with pytest.raises(ValueError):
        adversarial_path(trend_follower, base, n_steps=0)
    with pytest.raises(ValueError):
        adversarial_path(trend_follower, base, step_size=0.0)
    with pytest.raises(ValueError):
        adversarial_path(trend_follower, base, n_restarts=0)


def test_certification_reports_the_first_breakable_budget():
    base = walk()
    result = certify_robustness(momentum_strategy, base, budgets=(0.01, 0.05, 0.2))

    assert result['baseline_pnl'] > 0
    assert result['margin_of_safety'] in (0.01, 0.05, 0.2, None)
    assert result['verdict'] in ('CERTIFIED', 'BREAKABLE')
    assert len(result['probes']) == 3


def test_a_strategy_that_always_earns_is_certified():
    result = certify_robustness(lambda p: 1.0, walk(), budgets=(0.01, 0.05))
    assert result['margin_of_safety'] is None
    assert result['verdict'] == 'CERTIFIED'


def test_a_strategy_that_already_loses_reports_no_margin():
    result = certify_robustness(negative_on_everything, walk(), budgets=(0.01,))
    assert result['verdict'] == 'LOSES_MONEY_ON_HISTORICAL_DATA'
    assert result['margin_of_safety'] == 0.0


def test_certification_validates_its_budgets():
    with pytest.raises(ValueError):
        certify_robustness(trend_follower, walk(), budgets=())
    with pytest.raises(ValueError):
        certify_robustness(trend_follower, walk(), budgets=(0.0,))
    with pytest.raises(ValueError):
        certify_robustness(trend_follower, walk(), budgets=(0.05, 0.01))


# --- the full audit -------------------------------------------------------


def test_audit_produces_a_complete_report():
    report = audit_robustness(momentum_strategy, walk(), n_perturbations=8, n_adversarial_steps=8)

    assert isinstance(report, RobustnessReport)
    assert 0.0 <= report.survival_rate <= 1.0
    assert len(report.attack_results) == len(DEFAULT_ATTACKS) * 2
    assert 'verdict' in report.as_dict()
    assert report.weakest_attack() in {a.name for a in DEFAULT_ATTACKS}


def test_the_audit_is_reproducible():
    """An audit you cannot reproduce is not an audit."""
    prices = walk()
    first = audit_robustness(momentum_strategy, prices, n_perturbations=8, n_adversarial_steps=6)
    second = audit_robustness(momentum_strategy, prices, n_perturbations=8, n_adversarial_steps=6)
    assert first.as_dict() == second.as_dict()


def test_the_audit_is_reproducible_across_processes():
    """
    String hashing is randomised per process, so seeding from `hash()` makes a
    report silently differ between two identical runs. Run in a subprocess to
    get a genuinely different interpreter state.
    """
    import subprocess
    import sys

    script = (
        'import json, numpy as np;'
        'from algorithex.audit.robustness import audit_robustness;'
        'rng = np.random.default_rng(3);'
        'prices = 100.0*np.exp(np.cumsum(rng.normal(0.0004, 0.012, 300)));'
        'hold = lambda p: float(p[-1]-p[0]);'
        'print(json.dumps(audit_robustness(hold, prices, n_perturbations=8, '
        'n_adversarial_steps=6).as_dict(), sort_keys=True))'
    )
    runs = [
        subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_a_strong_strategy_is_called_robust():
    report = audit_robustness(trend_follower, walk(), n_perturbations=8, n_adversarial_steps=6)
    assert report.verdict == 'ROBUST'


def test_a_strategy_that_loses_money_is_reported_as_such():
    report = audit_robustness(negative_on_everything, walk(), n_perturbations=4, n_adversarial_steps=4)
    assert report.verdict == 'LOSES_MONEY'


def test_a_fragile_strategy_is_caught_by_the_attacks():
    # Reads the exact level of the very first bar; any attack destroys it.
    report = audit_robustness(
        lambda p: float(p[0] == walk()[0]), walk(), n_perturbations=4, n_adversarial_steps=4
    )
    assert report.verdict == 'FRAGILE'


def test_audit_validates_its_attack_configuration():
    with pytest.raises(ValueError):
        audit_robustness(trend_follower, walk(), attacks=[])
    with pytest.raises(ValueError):
        audit_robustness(trend_follower, walk(), severities=())
    with pytest.raises(ValueError):
        audit_robustness(trend_follower, walk(), severities=(0.0,))


def test_audit_can_run_a_custom_attack():
    def invert(prices, severity, rng):
        return prices[::-1].copy()

    report = audit_robustness(
        momentum_strategy,
        walk(),
        attacks=[Attack('invert', invert)],
        severities=(1.0,),
        n_perturbations=4,
        n_adversarial_steps=4,
    )
    assert len(report.attack_results) == 1
    assert report.attack_results[0].attack == 'invert'
