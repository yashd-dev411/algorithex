import numpy as np
import pytest

from algorithex.research.factor import (
    cornish_fisher_var,
    drawdown_attribution,
    expected_shortfall,
    factor_decomposition,
    historical_var,
    risk_attribution,
)


def linear_returns(seed=1, n=2000, scale=0.01):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n) * scale
    f2 = rng.normal(0, 1, n) * scale
    y = 2.0 * f1 - 1.0 * f2 + rng.normal(0, 0.005, n)
    return {'f1': f1, 'f2': f2}, y


def test_decomposition_recovers_known_loadings():
    factors, returns = linear_returns()

    model = factor_decomposition(factors, returns)

    assert model.loadings[0] == pytest.approx(2.0, rel=0.1)
    assert model.loadings[1] == pytest.approx(-1.0, rel=0.1)
    assert model.r_squared > 0.9


def test_decomposition_contributions_sum_to_the_mean_explained():
    factors, returns = linear_returns()

    model = factor_decomposition(factors, returns)
    explained = float(model.contributions.sum())

    # The intercept is excluded from contributions, so what remains is the part
    # the factors explain, and must be a minority of a noisy mean near zero.
    assert abs(explained) < abs(returns.mean()) + 1e-9
    assert set(model.as_dict()) == {'f1', 'f2', '_residual_std'}


def test_decomposition_handles_an_uninformative_factor():
    rng = np.random.default_rng(5)
    factors = {'signal': rng.normal(0, 0.01, 500), 'noise': rng.normal(0, 0.01, 500)}
    returns = 3.0 * factors['signal']

    model = factor_decomposition(factors, returns)

    assert abs(model.loadings[1]) < abs(model.loadings[0])


def test_risk_attribution_sums_to_portfolio_volatility():
    factors, returns = linear_returns()
    weights = np.array([2.0, -1.0])

    attribution = risk_attribution(factors, returns, weights=weights)

    covariance = np.cov(np.column_stack([factors['f1'], factors['f2']]), rowvar=False, ddof=1)
    total = float(np.sqrt(weights @ covariance @ weights))

    # Contributions are a *risk* decomposition, so they must sum to volatility.
    # Summing to the variance instead is a scale error that still looks plausible.
    assert sum(attribution.values()) == pytest.approx(total, rel=1e-9)


def test_risk_attribution_of_a_single_factor_is_weight_times_sigma():
    rng = np.random.default_rng(21)
    factor = rng.normal(0, 0.02, 500)
    returns = 3.0 * factor

    attribution = risk_attribution({'f': factor}, returns, weights=[3.0])

    assert attribution['f'] == pytest.approx(3.0 * float(np.std(factor, ddof=1)), rel=1e-9)


def test_risk_attribution_rejects_mismatched_weights():
    factors, returns = linear_returns()

    with pytest.raises(ValueError):
        risk_attribution(factors, returns, weights=[1.0])


def test_risk_attribution_is_positive_for_aligned_factors():
    rng = np.random.default_rng(7)
    factor = rng.normal(0, 0.01, 500)
    attribution = risk_attribution({'trend': factor}, 2.0 * factor)

    assert attribution['trend'] > 0


def test_expected_shortfall_exceeds_var_on_a_loss_tail():
    rng = np.random.default_rng(3)
    # Fat, negatively skewed tails, which is what real returns look like.
    returns = rng.standard_t(3, 4000) * 0.01 - 0.002

    var95 = historical_var(returns, 0.95)
    es95 = expected_shortfall(returns, 0.95)

    assert es95 >= var95


def test_cornish_fisher_reports_more_risk_than_gaussian_on_fat_tails():
    rng = np.random.default_rng(4)
    returns = rng.standard_t(4, 5000) * 0.01

    result = cornish_fisher_var(returns, 0.95)

    assert result['excess_kurtosis'] > 0
    assert result['cornish_fisher_var'] > result['gaussian_var']
    assert result['correction'] > 0


def test_cornish_fisher_is_zero_on_constant_returns():
    result = cornish_fisher_var(np.full(200, 0.001), 0.95)

    # A constant series has no dispersion, so every risk figure is zero and the
    # skew/kurtosis estimates are undefined rather than large.
    assert result['cornish_fisher_var'] == 0.0
    assert result['skew'] == 0.0
    assert result['excess_kurtosis'] == 0.0


def test_var_grows_with_confidence():
    rng = np.random.default_rng(9)
    returns = rng.normal(0, 0.01, 2000)

    assert historical_var(returns, 0.99) > historical_var(returns, 0.90)


def test_drawdown_attribution_locates_the_worst_period():
    rng = np.random.default_rng(11)
    n = 1000
    returns = np.full(n, 0.001)
    returns[400:420] = -0.05  # a clear, injected crash

    report = drawdown_attribution({'f': rng.normal(0, 0.001, n)}, returns)

    assert report['trough_index'] >= 400
    assert report['trough_index'] <= 425
    assert report['max_drawdown'] < -0.2
    assert 'contributions' in report


def test_drawdown_attribution_reports_recovery():
    rng = np.random.default_rng(12)
    n = 1500
    returns = np.full(n, 0.002)
    returns[100:120] = -0.04

    report = drawdown_attribution({'f': rng.normal(0, 0.001, n)}, returns)

    assert report['recovered'] is True
    assert report['recovery_index'] > report['trough_index']


def test_drawdown_attribution_reports_an_unrecovered_drawdown():
    rng = np.random.default_rng(22)
    n = 300
    returns = np.full(n, 0.002)
    returns[100:120] = -0.04

    # Too little time after the crash to climb back to the prior peak.
    report = drawdown_attribution({'f': rng.normal(0, 0.001, n)}, returns)

    assert report['recovered'] is False
    assert report['recovery_index'] is None


def test_drawdown_attribution_rejects_impossible_returns():
    rng = np.random.default_rng(13)
    factors = {'f': rng.normal(0, 0.01, 200)}
    returns = np.full(200, -2.0)  # a -200% "return" is not a return

    with pytest.raises(ValueError):
        drawdown_attribution(factors, returns)


def test_factor_helpers_reject_bad_input():
    rng = np.random.default_rng(14)
    returns = rng.normal(0, 0.01, 100)

    with pytest.raises(ValueError):
        factor_decomposition({}, returns)
    with pytest.raises(ValueError):
        factor_decomposition({'a': rng.normal(size=50)}, returns)
    with pytest.raises(ValueError):
        factor_decomposition({'a': returns[:10], 'b': returns[:20]}, returns)
    with pytest.raises(ValueError):
        historical_var(returns, 1.5)
    with pytest.raises(ValueError):
        expected_shortfall([np.nan, np.nan])
    with pytest.raises(ValueError):
        historical_var([])


def test_helpers_ignore_non_finite_observations():
    rng = np.random.default_rng(15)
    returns = rng.normal(0, 0.01, 300)
    dirty = returns.copy()
    dirty[:5] = np.nan

    assert np.isfinite(historical_var(dirty))
    assert np.isfinite(expected_shortfall(dirty))