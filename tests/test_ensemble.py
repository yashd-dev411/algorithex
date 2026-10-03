import numpy as np
import pytest

from algorithex.research.ensemble import (
    StrategyReturns,
    WeightPath,
    apply_regime_allocation,
    build_returns_matrix,
    drawdown_penalised_weights,
    effective_strategies,
    ensemble_returns,
    equal_weight_weights,
    evaluate_ensemble,
    exponential_weights,
    information_ratio_weights,
    inverse_volatility_weights,
    marginal_contribution,
    max_drawdown,
    positions_to_returns,
    regime_arbitration,
    walk_forward_ensemble,
    weight_stability,
)


def independent_returns(seed=5, n=400, n_strategies=3, vols=None):
    """Members with unrelated returns: the case where combining genuinely helps."""
    rng = np.random.default_rng(seed)
    vols = np.array(vols if vols is not None else [1.0, 2.0, 0.5][:n_strategies])
    return rng.normal(0.0, 1.0, (vols.size, n)) * vols[:, None]


def trend_then_chop(seed=9, n=600):
    """A trend follower and a mean reverter, each earning in its own regime."""
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(n) < n // 2, 0.004, -0.001)
    noise = rng.normal(0.0, 0.01, n)
    return np.vstack([drift + noise, -drift + noise])


def as_strategies(matrix, prefix='s'):
    return [
        StrategyReturns(f'{prefix}{i}', row) for i, row in enumerate(np.asarray(matrix))
    ]


# --- construction ---------------------------------------------------------


def test_returns_matrix_stacks_aligned_series():
    matrix = independent_returns()
    names, built = build_returns_matrix(as_strategies(matrix))

    assert names == ['s0', 's1', 's2']
    assert built.shape == matrix.shape


def test_misaligned_series_are_rejected_not_silently_padded():
    strategies = [
        StrategyReturns('a', np.zeros(10)),
        StrategyReturns('b', np.zeros(9)),
    ]
    with pytest.raises(ValueError, match='must be aligned'):
        build_returns_matrix(strategies)


def test_duplicate_and_empty_and_non_finite_inputs_are_rejected():
    with pytest.raises(ValueError):
        build_returns_matrix([])
    with pytest.raises(ValueError, match='duplicate'):
        build_returns_matrix(
            [StrategyReturns('a', np.zeros(5)), StrategyReturns('a', np.zeros(5))]
        )
    with pytest.raises(ValueError, match='non-finite'):
        StrategyReturns('a', np.array([1.0, np.nan]))
    with pytest.raises(TypeError):
        build_returns_matrix([np.zeros(5)])


def test_positions_to_returns_captures_the_move_and_pays_turnover():
    prices = np.array([100.0, 110.0, 121.0])
    positions = np.array([1.0, 1.0, 0.0])

    result = positions_to_returns(positions, prices)
    assert result[0] == pytest.approx(0.10)
    assert result[1] == pytest.approx(0.10)

    # Entering the position costs a fee; simply holding it does not.
    with_fee = positions_to_returns(positions, prices, fee=0.001)
    assert with_fee[0] == pytest.approx(0.099)
    assert with_fee[1] == pytest.approx(0.10)


def test_positions_to_returns_validates_its_inputs():
    with pytest.raises(ValueError):
        positions_to_returns(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        positions_to_returns(np.ones(3), np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        positions_to_returns(np.array([1.0]), np.array([1.0]))


# --- allocation schemes ---------------------------------------------------


def test_equal_weight_is_uniform():
    assert np.allclose(equal_weight_weights(4), 0.25)
    with pytest.raises(ValueError):
        equal_weight_weights(0)


def test_inverse_volatility_favours_the_calmer_member():
    matrix = independent_returns(vols=[1.0, 4.0])
    weights = inverse_volatility_weights(matrix)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] * 3


def test_inverse_volatility_keeps_a_constant_member_in_the_book():
    """Zero volatility must not silently delete a strategy."""
    matrix = np.vstack([np.zeros(50), np.random.default_rng(1).normal(0, 1, 50)])
    weights = inverse_volatility_weights(matrix)

    assert weights[0] > 0
    assert weights.sum() == pytest.approx(1.0)


def test_weight_cap_redistributes_rather_than_clipping():
    weights = inverse_volatility_weights(
        np.vstack([np.full(60, 0.001), np.random.default_rng(2).normal(0, 1, 60)]),
        max_weight=0.8,
    )
    assert weights.max() <= 0.8 + 1e-9
    assert weights.sum() == pytest.approx(1.0)


def test_an_impossible_weight_cap_is_rejected():
    matrix = independent_returns(n_strategies=5)
    with pytest.raises(ValueError, match='cannot be honoured'):
        inverse_volatility_weights(matrix, max_weight=0.1)


def test_exponential_weights_chase_the_member_that_earned():
    # s0 earns steadily, s1 loses steadily.
    matrix = np.vstack([np.full(50, 0.01), np.full(50, -0.01)])
    weights = exponential_weights(matrix, learning_rate=1.0)

    assert weights[0] > weights[1]
    assert weights.sum() == pytest.approx(1.0)


def test_exponential_weights_at_zero_learning_rate_is_equal_weight():
    matrix = independent_returns()
    weights = exponential_weights(matrix, learning_rate=0.0)
    assert np.allclose(weights, equal_weight_weights(matrix.shape[0]))


def test_exponential_weights_survive_an_extreme_bar():
    """A -900% bar must not drive a member's weight to exactly zero forever."""
    matrix = np.vstack([np.full(50, 0.001), np.full(50, 0.001)])
    matrix[1, 10] = -9.0
    weights = exponential_weights(matrix, learning_rate=2.0)

    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1.0)


def test_exponential_weights_validate_learning_rate():
    with pytest.raises(ValueError):
        exponential_weights(independent_returns(), learning_rate=-1.0)
    with pytest.raises(ValueError):
        exponential_weights(independent_returns(), learning_rate=np.nan)


def test_drawdown_penalty_disfavours_the_strategy_that_halved():
    steady = np.full(200, 0.002)
    crashed = np.concatenate([np.full(100, 0.02), np.full(100, -0.02)])
    matrix = np.vstack([steady, crashed])

    weights = drawdown_penalised_weights(matrix, strength=2.0)
    assert weights[0] > weights[1]


def test_max_drawdown_is_measured_on_the_equity_curve():
    # No single bar is catastrophic, but the path bleeds steadily.
    bleed = -np.full(100, 0.02)
    assert max_drawdown(bleed) > 0.5
    assert max_drawdown(np.zeros(10)) == 0.0
    assert max_drawdown(np.array([])) == 0.0


# --- combining ------------------------------------------------------------


def test_ensemble_returns_accepts_a_static_or_a_per_bar_allocation():
    matrix = independent_returns(n=50)
    static = ensemble_returns(matrix, equal_weight_weights(3))
    assert static.shape == (50,)

    path = np.tile(equal_weight_weights(3), (50, 1))
    assert np.allclose(ensemble_returns(matrix, path), static)


def test_ensemble_returns_rejects_mismatched_shapes():
    matrix = independent_returns(n=50)
    with pytest.raises(ValueError):
        ensemble_returns(matrix, np.ones(7))
    with pytest.raises(ValueError):
        ensemble_returns(matrix, np.ones((50, 7)))
    with pytest.raises(ValueError):
        ensemble_returns(matrix, np.ones((40, 3)))
    with pytest.raises(ValueError):
        ensemble_returns(matrix, np.ones((2, 2, 3)))


def test_diversifying_independent_members_beats_either_alone():
    """The entire argument for running an ensemble."""
    matrix = independent_returns(seed=5, n=800)
    ensemble = ensemble_returns(matrix, equal_weight_weights(3))
    ensemble_sharpe = ensemble.mean() / ensemble.std(ddof=1)
    member_sharpes = [row.mean() / row.std(ddof=1) for row in matrix]

    # Averaging uncorrelated streams cuts variance without cutting mean.
    assert ensemble.std(ddof=1) < matrix.std(axis=1, ddof=1).mean()
    assert ensemble_sharpe >= max(member_sharpes) * 0.5


def test_effective_strategies_detects_a_collapsed_ensemble():
    assert effective_strategies(np.full(5, 0.2)) == pytest.approx(5.0, abs=0.01)
    assert effective_strategies(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert effective_strategies(np.array([9.0, 0.5, 0.5])) < 1.3

    with pytest.raises(ValueError):
        effective_strategies(np.zeros(3))


# --- regime arbitration ---------------------------------------------------


def test_regime_arbitration_allocates_differently_per_regime():
    matrix = trend_then_chop()
    labels = np.array([0] * 300 + [1] * 300)
    allocations, _ = regime_arbitration(matrix, labels, shrinkage=0.0, min_obs=50)

    assert set(allocations) == {0, 1}
    # In the trending half the trend follower should outweigh the reverter.
    assert allocations[0][0] > allocations[0][1]
    assert allocations[1][1] > allocations[1][0]


def test_regime_arbitration_shrinks_towards_the_global_allocation():
    """Unshrunk per-regime weights are how alphas get invented."""
    matrix = trend_then_chop()
    labels = np.array([0] * 300 + [1] * 300)
    loose, _ = regime_arbitration(matrix, labels, shrinkage=0.0, min_obs=50)
    tight, _ = regime_arbitration(matrix, labels, shrinkage=1.0, min_obs=50)

    # Full shrinkage collapses every regime onto the same global allocation.
    assert np.allclose(tight[0], tight[1])
    assert not np.allclose(loose[0], loose[1])


def test_a_thin_regime_refuses_to_specialise():
    matrix = trend_then_chop()
    labels = np.array([0] * 597 + [1] * 3)
    allocations, _ = regime_arbitration(matrix, labels, shrinkage=0.0, min_obs=30)
    global_weights = inverse_volatility_weights(matrix)

    assert np.allclose(allocations[1], global_weights)


def test_information_ratio_allocation_funds_the_earner():
    winner = np.full(300, 0.004) + np.random.default_rng(1).normal(0, 0.01, 300)
    loser = np.full(300, -0.004) + np.random.default_rng(2).normal(0, 0.01, 300)
    weights = information_ratio_weights(np.vstack([winner, loser]))

    assert weights[0] > 0.99
    assert weights[1] == pytest.approx(0.0, abs=1e-9)


def test_information_ratio_falls_back_when_nobody_earned():
    dead = np.zeros((3, 200))
    assert np.allclose(information_ratio_weights(dead), equal_weight_weights(3))


def test_information_ratio_needs_enough_bars():
    with pytest.raises(ValueError, match='at least 2 bars'):
        information_ratio_weights(np.zeros((3, 1)))


def test_regime_arbitration_validates_its_inputs():
    matrix = trend_then_chop()
    with pytest.raises(ValueError, match='labels must be'):
        regime_arbitration(matrix, np.zeros(10))
    with pytest.raises(ValueError):
        regime_arbitration(matrix, np.array([0] * 300 + [1] * 300), shrinkage=2.0)
    with pytest.raises(ValueError):
        regime_arbitration(matrix, np.array([0] * 300 + [1] * 300), min_obs=0)


def test_an_unseen_regime_degrades_to_the_fallback():
    allocations = {0: np.array([0.8, 0.2])}
    path, unknown = apply_regime_allocation(allocations, np.array([0, 0, 7]))

    assert np.allclose(path[0], allocations[0])
    assert np.allclose(path[2], [0.5, 0.5])
    assert list(unknown) == [False, False, True]


def test_apply_regime_allocation_validates_its_inputs():
    with pytest.raises(ValueError):
        apply_regime_allocation({}, np.array([0]))
    with pytest.raises(ValueError):
        apply_regime_allocation({0: np.ones(2)}, np.array([[0, 1]]))
    with pytest.raises(ValueError, match='fallback has'):
        apply_regime_allocation({0: np.ones(2)}, np.array([0]), fallback=np.ones(3))


# --- causal refitting -----------------------------------------------------


@pytest.mark.parametrize(
    'scheme', ['equal_weight', 'inverse_volatility', 'exponential', 'drawdown']
)
def test_walk_forward_produces_a_valid_path_for_every_scheme(scheme):
    matrix = independent_returns(n=120)
    path = walk_forward_ensemble(matrix, scheme=scheme, train_window=40)

    assert isinstance(path, WeightPath)
    assert path.matrix.shape == (120, 3)
    assert np.allclose(path.matrix.sum(axis=1), 1.0)
    assert np.all(path.matrix >= -1e-12)


def test_walk_forward_refits_only_on_the_trailing_window():
    """A weight computed from the future is a label, not a weight."""
    matrix = independent_returns(n=100)
    short = walk_forward_ensemble(matrix, scheme='inverse_volatility', train_window=20).final
    long = walk_forward_ensemble(matrix, scheme='inverse_volatility', train_window=80).final

    assert not np.allclose(short, long)


def test_walk_forward_does_not_peek_at_the_bar_being_predicted():
    matrix = independent_returns(n=100)
    baseline = walk_forward_ensemble(matrix, scheme='exponential', train_window=30).matrix

    # Rewriting the final bar must not change the weights that produced it.
    tampered = matrix.copy()
    tampered[:, -1] = 5.0
    after = walk_forward_ensemble(tampered, scheme='exponential', train_window=30).matrix

    assert np.allclose(baseline[:-1], after[:-1])


def test_walk_forward_regime_scheme_uses_labels():
    matrix = trend_then_chop()
    labels = np.array([0] * 300 + [1] * 300)
    path = walk_forward_ensemble(
        matrix, scheme='regime', train_window=150, labels=labels, min_obs=40
    )
    assert path.matrix.shape == (600, 2)

    with pytest.raises(ValueError, match='needs a labels argument'):
        walk_forward_ensemble(matrix, scheme='regime')


def test_walk_forward_rejects_bad_configuration():
    matrix = independent_returns(n=50)
    with pytest.raises(ValueError, match='train_window'):
        walk_forward_ensemble(matrix, train_window=1)
    with pytest.raises(ValueError, match='unknown allocation scheme'):
        walk_forward_ensemble(matrix, scheme='vibes')


def test_weight_path_validates_its_shape():
    with pytest.raises(ValueError, match='columns'):
        WeightPath(np.zeros((10, 3)), ['a', 'b'])
    with pytest.raises(ValueError, match='non-finite'):
        WeightPath(np.full((10, 2), np.nan), ['a', 'b'])
    with pytest.raises(ValueError):
        WeightPath(np.zeros((0, 2)), ['a', 'b']).final


def test_weight_stability_distinguishes_static_from_churn():
    static = np.tile(np.array([0.5, 0.5]), (20, 1))
    churning = np.tile(np.array([0.9, 0.1]), (20, 1))
    churning[::2] = [0.1, 0.9]

    assert weight_stability(static) == 0.0
    assert weight_stability(churning) > 0.3
    assert weight_stability(np.zeros((1, 2))) == 0.0


# --- reporting ------------------------------------------------------------


def test_marginal_contribution_exposes_a_harmful_member():
    good = np.random.default_rng(3).normal(0.001, 0.01, 400)
    harmful = np.random.default_rng(4).normal(-0.01, 0.01, 400)
    matrix = np.vstack([good, harmful])

    contributions = marginal_contribution(matrix, np.array([0.5, 0.5]))
    by_index = {c['index']: c for c in contributions}

    assert by_index[0]['delta_sharpe'] > 0
    assert by_index[1]['delta_sharpe'] < 0


def test_evaluate_ensemble_reports_members_and_contributions():
    matrix = trend_then_chop()
    report = evaluate_ensemble(as_strategies(matrix))

    assert report.names == ['s0', 's1']
    assert set(report.members) == {'s0', 's1'}
    assert report.ensemble['observations'] == matrix.shape[1]
    assert len(report.contributions) == 2
    assert report.meta['weight_source'] == 'static'
    assert report.as_dict()['names'] == ['s0', 's1']


def test_evaluate_ensemble_flags_members_that_hurt():
    matrix = np.vstack(
        [
            np.random.default_rng(3).normal(0.002, 0.01, 500),
            np.random.default_rng(4).normal(-0.02, 0.01, 500),
        ]
    )
    report = evaluate_ensemble(as_strategies(matrix))
    assert 's1' in report.harmful_members()


def test_evaluate_ensemble_accepts_an_explicit_weight_path():
    matrix = independent_returns(n=200)
    path = np.tile(equal_weight_weights(3), (200, 1))
    report = evaluate_ensemble(as_strategies(matrix), weight_path=path)

    assert report.meta['weight_source'] == 'path'
    assert report.weight_stability == 0.0
    assert report.effective_strategies == pytest.approx(3.0, abs=0.01)


def test_walk_forward_evaluation_reports_turnover():
    matrix = trend_then_chop()
    path = walk_forward_ensemble(matrix, scheme='inverse_volatility', train_window=100)
    report = evaluate_ensemble(as_strategies(matrix), weight_path=path.matrix)

    assert report.weight_stability > 0
    assert 1.0 <= report.effective_strategies <= 2.0 + 1e-9


def test_report_can_be_asked_whether_combining_was_worth_it():
    matrix = independent_returns(seed=17, n=900)
    report = evaluate_ensemble(as_strategies(matrix))
    assert isinstance(report.beats_best_member(), bool)
    assert report.best_member() in report.members


def test_an_empty_report_cannot_be_ranked():
    from algorithex.research.ensemble import EnsembleReport

    report = EnsembleReport(
        names=[], weights=np.array([]), ensemble={}, members={},
        contributions=[], weight_stability=0.0, effective_strategies=0.0,
    )
    with pytest.raises(ValueError):
        report.best_member()
