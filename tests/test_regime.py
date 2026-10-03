import numpy as np
import pytest

from algorithex.research.regime import (
    GaussianHMM,
    best_split,
    detect_change_points,
    drawdown_throttle,
    realized_volatility,
    regime_position_size,
    regime_volatility,
    segment_cost,
    segments,
    volatility_target_size,
)


def two_regime_series(seed=7, n=1200):
    """A quiet half and a wild half that share a mean, differing only in spread."""
    rng = np.random.default_rng(seed)
    states = np.array([0] * (n // 2) + [1] * (n // 2))
    values = np.where(
        states[:, None] == 0, rng.normal(0, 0.01, (n, 1)), rng.normal(0, 0.05, (n, 1))
    )
    return values, states


def test_hmm_recovers_regimes_that_differ_only_in_volatility():
    values, states = two_regime_series()

    model = GaussianHMM(n_states=2, random_state=1).fit(values)
    predicted = model.predict(values)

    # State labels are arbitrary, so accept either orientation.
    agreement = max((predicted == states).mean(), ((1 - predicted) == states).mean())
    assert agreement > 0.95


def test_hmm_recovers_the_regime_volatilities():
    values, _ = two_regime_series()

    model = GaussianHMM(n_states=2, random_state=1).fit(values)
    recovered = np.sort(np.sqrt(model.covars_).ravel())

    assert recovered[0] == pytest.approx(0.01, rel=0.3)
    assert recovered[1] == pytest.approx(0.05, rel=0.3)


def test_hmm_likelihood_matches_explicit_enumeration():
    """Guards the forward algorithm against a subtly wrong log-normalisation."""
    from itertools import product

    rng = np.random.default_rng(3)
    n_states, n_samples, n_features = 2, 5, 2
    values = rng.normal(size=(n_samples, n_features)) + 0.4

    model = GaussianHMM(n_states, random_state=5)
    model.n_features_ = n_features
    model.means_ = rng.normal(size=(n_states, n_features))
    model.covars_ = np.array([rng.uniform(0.3, 1.2, n_features) for _ in range(n_states)])
    raw = rng.uniform(0.2, 1.0, (n_states, n_states))
    model.transmat_ = raw / raw.sum(axis=1, keepdims=True)
    start = rng.uniform(0.5, 1.0, n_states)
    model.startprob_ = start / start.sum()

    total = 0.0
    for path in product(range(n_states), repeat=n_samples):
        probability = model.startprob_[path[0]]
        for t in range(1, n_samples):
            probability *= model.transmat_[path[t - 1], path[t]]
        for t in range(n_samples):
            delta = values[t] - model.means_[path[t]]
            probability *= np.prod(
                1.0 / np.sqrt(2 * np.pi * model.covars_[path[t]])
            ) * np.exp(-0.5 * np.sum(delta ** 2 / model.covars_[path[t]]))
        total += probability

    assert model.score(values) == pytest.approx(np.log(total), abs=1e-8)


def test_hmm_posteriors_are_a_distribution():
    values, _ = two_regime_series(n=200)
    model = GaussianHMM(n_states=2, random_state=0).fit(values)

    posterior = model.predict_proba(values)

    assert posterior.shape == (200, 2)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.all(posterior >= 0.0)


def test_hmm_fit_is_reproducible_for_a_fixed_seed():
    values, _ = two_regime_series(n=200)

    first = GaussianHMM(2, random_state=7).fit(values).predict(values)
    second = GaussianHMM(2, random_state=7).fit(values).predict(values)

    assert np.array_equal(first, second)


def test_inline_logsumexp_matches_scipy():
    """The HMM uses a hand-rolled log-sum-exp for speed. It must be
    numerically indistinguishable from the SciPy function it replaces."""
    from scipy.special import logsumexp as reference

    from algorithex.research.regime.markov import _logsumexp

    rng = np.random.default_rng(0)
    for shape, axis in [((5,), None), ((7, 3), 0), ((7, 3), 1), ((4, 2, 3), 2)]:
        values = rng.normal(size=shape)
        assert np.allclose(
            _logsumexp(values, axis=axis), reference(values, axis=axis)
        )

    values = rng.normal(size=(7, 3))
    assert _logsumexp(values, axis=1, keepdims=True).shape == (7, 1)
    assert np.allclose(
        _logsumexp(values, axis=1, keepdims=True),
        reference(values, axis=1, keepdims=True),
    )


def test_inline_logsumexp_handles_all_negative_infinity():
    from scipy.special import logsumexp as reference

    from algorithex.research.regime.markov import _logsumexp

    values = np.full((3, 3), -np.inf)

    assert np.allclose(_logsumexp(values, axis=1), reference(values, axis=1))


def test_hmm_rejects_bad_input():
    with pytest.raises(ValueError):
        GaussianHMM(n_states=0)
    with pytest.raises(ValueError):
        GaussianHMM(2).fit([[1.0, np.nan]])
    with pytest.raises(ValueError):
        GaussianHMM(2).fit([[]])
    with pytest.raises(RuntimeError):
        GaussianHMM(2).predict(np.zeros((10, 1)))


def test_hmm_handles_a_single_state_and_a_single_observation():
    values, _ = two_regime_series(n=100)

    assert GaussianHMM(1, random_state=0).fit(values).predict(values).shape == (100,)
    assert GaussianHMM(2, random_state=0).fit(np.zeros((1, 2))).predict(np.zeros((1, 2))).shape == (1,)


def test_segment_cost_prefers_homogeneous_segments():
    steady = np.full(50, 3.0)
    mixed = np.concatenate([np.zeros(25), np.full(25, 50.0)])

    assert segment_cost(steady) < segment_cost(mixed)


def test_best_split_finds_the_boundary():
    values = np.concatenate([np.full(50, 0.0), np.full(50, 10.0)])

    split, _ = best_split(values, min_segment=5)

    assert split == 50


def test_detect_change_points_finds_a_known_shift():
    rng = np.random.default_rng(11)
    values = np.concatenate([rng.normal(0, 1, 400), rng.normal(3, 1, 400)])

    found = detect_change_points(values, min_segment=20)

    assert [c.index for c in found] == [400]


def test_detect_change_points_finds_multiple_shifts():
    rng = np.random.default_rng(11)
    values = np.concatenate([
        rng.normal(0, 1, 300), rng.normal(1, 1, 300), rng.normal(2.2, 1, 300),
    ])

    found = [c.index for c in detect_change_points(values, min_segment=20)]

    assert len(found) == 2
    # Locate the breakpoint to within sampling noise rather than to the bar:
    # the true boundaries are 300 and 600, and the estimates carry a few bars
    # of uncertainty in both directions.
    assert abs(found[0] - 300) <= 20
    assert abs(found[1] - 600) <= 20


def test_detect_change_points_ignores_stationary_noise():
    rng = np.random.default_rng(11)

    assert detect_change_points(rng.normal(0, 1, 800), min_segment=20) == []


def test_detect_change_points_respects_the_cap():
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0, 1, 200), rng.normal(4, 1, 200)])

    capped = detect_change_points(values, min_segment=20, n_points=1)

    assert len(capped) == 1


def test_detect_change_points_rejects_bad_input():
    with pytest.raises(ValueError):
        detect_change_points(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        detect_change_points([1.0, np.nan, 2.0])
    assert detect_change_points(np.zeros(5), min_segment=10) == []


def test_segments_split_at_the_detected_points():
    rng = np.random.default_rng(11)
    values = np.concatenate([rng.normal(0, 1, 400), rng.normal(3, 1, 400)])

    parts = segments(values, detect_change_points(values, min_segment=20))

    assert parts == [(0, 400), (400, 800)]


def test_realized_volatility_leaves_the_warmup_unset():
    returns = np.random.default_rng(0).normal(0, 0.01, 100)

    values = realized_volatility(returns, window=30)

    assert np.isnan(values[:29]).all()
    assert np.all(np.isfinite(values[29:]))


def test_volatility_target_size_inverses_observed_volatility():
    observed = np.array([0.02, 0.01])

    sizes = volatility_target_size(observed, target_vol=0.01, upper=5.0)

    assert sizes[0] == pytest.approx(0.5)
    assert sizes[1] == pytest.approx(1.0)


def test_volatility_target_size_respects_bounds_and_passes_nan_through():
    sizes = volatility_target_size(
        np.array([np.nan, 0.001]), target_vol=0.01, lower=0.0, upper=1.0
    )

    assert np.isnan(sizes[0])
    assert sizes[1] == pytest.approx(1.0)


def test_regime_volatility_separates_calm_from_wild():
    rng = np.random.default_rng(2)
    states = np.array([0] * 150 + [1] * 150)
    calm = rng.normal(0, 0.005, 150)
    wild = rng.normal(0, 0.03, 150)
    # The regimes are contiguous halves, so the series is a concatenation.
    returns = np.concatenate([calm, wild])

    summary = regime_volatility(states, returns)

    assert summary[1]['volatility'] > summary[0]['volatility'] * 3
    assert summary['_spread']['ratio'] > 3.0


def test_regime_position_size_is_smaller_in_the_wild_regime():
    rng = np.random.default_rng(1)
    states = np.array([0] * 150 + [1] * 150)
    returns = np.where(states == 0, rng.normal(0, 0.005, 300), rng.normal(0, 0.03, 300))

    sizes = regime_position_size(states, returns, target_vol=0.01, window=30)

    assert sizes[140] > sizes[290]


def test_regime_position_size_falls_back_to_warmup():
    states = np.zeros(10, dtype=int)
    returns = np.random.default_rng(0).normal(0, 0.01, 10)

    sizes = regime_position_size(states, returns, window=30, warmup_state=0.5)

    assert np.allclose(sizes, 0.5)


def test_drawdown_throttle_reduces_then_restores_size():
    equity = np.concatenate([
        np.linspace(1.0, 2.0, 200),
        np.linspace(2.0, 1.2, 200),
        np.linspace(1.2, 1.9, 200),
    ])

    scale = drawdown_throttle(equity, max_drawdown=0.2, window=100)

    assert scale[0] == pytest.approx(1.0)
    assert scale[350] < scale[0]
    assert scale[-1] > scale[350]


def test_drawdown_throttle_without_recovery_stays_down():
    equity = np.concatenate([np.linspace(1.0, 2.0, 200), np.linspace(2.0, 1.0, 200)])

    scale = drawdown_throttle(equity, max_drawdown=0.2, recovery=False)

    assert scale[-1] == pytest.approx(0.0)


def test_sizing_helpers_reject_invalid_arguments():
    with pytest.raises(ValueError):
        realized_volatility([1.0, 2.0], window=1)
    with pytest.raises(ValueError):
        volatility_target_size([0.01], target_vol=0.0)
    with pytest.raises(ValueError):
        regime_volatility([0, 1, 0], [1.0, 2.0])
    with pytest.raises(ValueError):
        drawdown_throttle([1.0], max_drawdown=0.0)
    with pytest.raises(ValueError):
        drawdown_throttle([], max_drawdown=0.2)