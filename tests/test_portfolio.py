import numpy as np
import pytest

from algorithex.portfolio import (
    constant_correlation_covariance,
    diversification_ratio,
    equal_weight_weights,
    minimum_variance_weights,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
    shrunk_covariance,
)


def correlated_returns(seed=3, n=600):
    """Three assets at known volatilities, driven by a shared factor."""
    rng = np.random.default_rng(seed)
    target = np.array([0.05, 0.15, 0.30])
    loadings = rng.normal(size=(3, 3))
    values = rng.normal(size=(n, 3)) @ loadings
    return values / values.std(axis=0) * target


def risk_shares(weights, covariance):
    contributions = risk_contributions(weights, covariance)
    return contributions / contributions.sum()


def test_constant_correlation_matrix_has_the_requested_structure():
    volatilities = np.array([0.1, 0.2])
    matrix = constant_correlation_covariance(volatilities, correlation=0.5)

    assert matrix[0, 0] == pytest.approx(0.01)
    assert matrix[0, 1] == pytest.approx(0.5 * 0.1 * 0.2)
    assert np.allclose(matrix, matrix.T)


def test_constant_correlation_rejects_impossible_correlations():
    with pytest.raises(ValueError):
        constant_correlation_covariance([0.1, 0.2], correlation=1.5)


def test_shrunk_covariance_preserves_volatility_and_damps_correlation():
    values = correlated_returns()
    sample = np.cov(values, rowvar=False)

    shrunk = shrunk_covariance(values, shrinkage=0.8, correlation=0.0)

    # With a zero-correlation target, off-diagonals shrink toward zero.
    assert abs(shrunk[0, 1]) < abs(sample[0, 1])
    # Volatilities are preserved by construction.
    assert np.allclose(np.sqrt(np.diag(shrunk)), np.sqrt(np.diag(sample)), rtol=0.15)


def test_shrunk_covariance_is_positive_definite_with_few_observations():
    values = correlated_returns(n=6)

    shrunk = shrunk_covariance(values, shrinkage=0.5)

    assert np.all(np.linalg.eigvalsh(shrunk) > 0)


def test_shrunk_covariance_rejects_bad_input():
    with pytest.raises(ValueError):
        shrunk_covariance(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        shrunk_covariance(correlated_returns(), shrinkage=2.0)


def test_equal_weights_sum_to_one():
    weights = equal_weight_weights(4)

    assert np.allclose(weights, 0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_minimum_variance_beats_equal_weight_on_its_own_objective():
    covariance = shrunk_covariance(correlated_returns())

    equal = portfolio_volatility(equal_weight_weights(3), covariance) ** 2
    optimised = portfolio_volatility(minimum_variance_weights(covariance), covariance) ** 2

    assert optimised < equal


def test_minimum_variance_matches_the_closed_form_for_uncorrelated_assets():
    # Unconstrained minimum variance puts weight proportional to 1 / variance.
    volatilities = np.array([0.05, 0.15, 0.30])
    covariance = np.diag(volatilities ** 2)
    expected = 1.0 / volatilities ** 2
    expected /= expected.sum()

    assert np.allclose(minimum_variance_weights(covariance), expected, atol=1e-6)


def test_weight_cap_is_respected_and_actually_binds():
    covariance = shrunk_covariance(correlated_returns())

    uncapped = minimum_variance_weights(covariance)
    capped = minimum_variance_weights(covariance, max_weight=0.4)

    assert uncapped.max() > 0.4
    assert capped.max() <= 0.4 + 1e-9
    assert capped.sum() == pytest.approx(1.0)
    # The cap must cost something; that is the whole point of imposing it.
    assert portfolio_volatility(capped, covariance) ** 2 >= (
        portfolio_volatility(uncapped, covariance) ** 2 - 1e-9
    )


def test_infeasible_cap_is_rejected():
    covariance = np.eye(4)

    with pytest.raises(ValueError):
        minimum_variance_weights(covariance, max_weight=0.2)


def test_risk_parity_equals_inverse_volatility_for_uncorrelated_assets():
    volatilities = np.array([0.05, 0.15, 0.30])
    covariance = np.diag(volatilities ** 2)
    expected = 1.0 / volatilities
    expected /= expected.sum()

    weights = risk_parity_weights(covariance)

    assert np.allclose(weights, expected, atol=1e-4)


def test_risk_parity_actually_equalises_risk_contributions():
    """The defining property. A solver that returns plausible weights but does
    not equalise contributions would pass a weight-sum check and fail here."""
    covariance = shrunk_covariance(correlated_returns())

    shares = risk_shares(risk_parity_weights(covariance), covariance)

    assert np.allclose(shares, 1.0 / 3, atol=1e-4)


def test_risk_parity_diversifies_better_than_equal_weight():
    covariance = shrunk_covariance(correlated_returns())

    equal = diversification_ratio(equal_weight_weights(3), covariance)
    parity = diversification_ratio(risk_parity_weights(covariance), covariance)

    assert parity > equal


def test_risk_parity_honours_custom_risk_budgets():
    covariance = shrunk_covariance(correlated_returns())
    budgets = np.array([0.5, 0.3, 0.2])

    shares = risk_shares(risk_parity_weights(covariance, budgets), covariance)

    assert np.allclose(shares, budgets, atol=1e-4)


def test_risk_parity_is_reproducible():
    covariance = shrunk_covariance(correlated_returns())

    first = risk_parity_weights(covariance, random_state=5)
    second = risk_parity_weights(covariance, random_state=5)

    assert np.allclose(first, second)


def test_risk_parity_rejects_bad_budgets():
    covariance = np.eye(3)

    with pytest.raises(ValueError):
        risk_parity_weights(covariance, target_contributions=[0.5, 0.5])
    with pytest.raises(ValueError):
        risk_parity_weights(covariance, target_contributions=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        risk_parity_weights(np.zeros((3, 3)))


def test_risk_contributions_sum_to_portfolio_volatility():
    covariance = shrunk_covariance(correlated_returns())
    weights = minimum_variance_weights(covariance)

    contributions = risk_contributions(weights, covariance)

    assert contributions.sum() == pytest.approx(portfolio_volatility(weights, covariance))


def test_diversification_ratio_is_one_for_a_single_asset():
    covariance = np.array([[0.04]])

    assert diversification_ratio(np.array([1.0]), covariance) == pytest.approx(1.0)


def test_diagnostics_reject_mismatched_dimensions():
    with pytest.raises(ValueError):
        portfolio_volatility([0.5, 0.5], np.eye(3))
    with pytest.raises(ValueError):
        risk_contributions([0.5, 0.5], np.eye(3))
    with pytest.raises(ValueError):
        portfolio_volatility([1.0], np.ones((2, 2)))