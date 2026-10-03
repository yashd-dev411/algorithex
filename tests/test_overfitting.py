import numpy as np
import pytest

from algorithex.research.overfitting import (
    DeflatedSharpeResult,
    PboResult,
    PlateauResult,
    deflated_sharpe,
    expected_max_sharpe,
    overfitting_report,
    plateau_analysis,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def noisy_matrix(seed=11, n_configs=40, n_bars=800, alphas=None):
    """
    Configurations whose true quality varies, all measured on independent noise.

    `alphas` is the per-configuration edge. Passing all zeros gives pure noise,
    which is the case where in-sample ranking should carry no information.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array(alphas if alphas is not None else np.zeros(n_configs))
    return rng.normal(0.0, 1.0, (n_configs, n_bars)) * 0.01 + alphas[:, None]


def trend_matrix(seed=12, n_configs=20, n_bars=800):
    """One configuration genuinely works; the rest are noise. PBO should be low."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.01, (n_configs, n_bars))
    matrix[0] += 0.004
    return matrix


# --- sharpe ---------------------------------------------------------------


def test_sharpe_is_annualised_and_scale_invariant():
    base = np.random.default_rng(1).normal(0.001, 0.01, 500)
    assert sharpe_ratio(base) == pytest.approx(
        sharpe_ratio(base * 3.0), rel=1e-9
    )
    assert sharpe_ratio(base) == pytest.approx(
        base.mean() / base.std() * np.sqrt(365)
    )


def test_sharpe_handles_degenerate_input():
    assert np.isnan(sharpe_ratio(np.array([1.0])))
    assert sharpe_ratio(np.zeros(50)) == float('-inf')
    assert sharpe_ratio(np.full(50, 0.1)) == float('inf')


# --- PBO ------------------------------------------------------------------


def test_pure_noise_has_a_pbo_near_one_half():
    """If nothing works, in-sample ranking must carry no information."""
    result = probability_of_backtest_overfitting(noisy_matrix(), n_blocks=8)
    assert 0.3 < result.pbo < 0.7
    assert result.n_configurations == 40
    assert result.n_splits > 0


def test_a_genuine_edge_transfers_out_of_sample():
    result = probability_of_backtest_overfitting(trend_matrix(), n_blocks=8)
    assert result.pbo < 0.5
    assert result.verdict == 'UNLIKELY_TO_BE_OVERFIT'
    assert not result.concerning


def test_pbo_grows_when_the_real_edge_is_buried_in_noise():
    weak = probability_of_backtest_overfitting(
        noisy_matrix(seed=4, n_configs=30, alphas=np.r_[0.0004, np.zeros(29)]), n_blocks=8
    ).pbo
    strong = probability_of_backtest_overfitting(
        noisy_matrix(seed=4, n_configs=30, alphas=np.r_[0.006, np.zeros(29)]), n_blocks=8
    ).pbo
    assert strong < weak


def test_pbo_reports_a_meaningful_logit_and_degradation():
    result = probability_of_backtest_overfitting(trend_matrix(), n_blocks=8)
    assert np.isfinite(result.logit)
    assert np.isfinite(result.performance_degradation)
    assert len(result.verdicts) == result.n_splits


def test_pbo_is_bounded_and_validates_its_blocks():
    with pytest.raises(ValueError, match='even integer'):
        probability_of_backtest_overfitting(noisy_matrix(), n_blocks=7)
    with pytest.raises(ValueError, match='cannot split'):
        probability_of_backtest_overfitting(noisy_matrix(n_bars=5), n_blocks=8)
    with pytest.raises(ValueError, match='at least 2 configurations'):
        probability_of_backtest_overfitting(noisy_matrix(n_configs=1))


def test_pbo_raises_when_no_split_is_usable():
    """Constant returns give infinite Sharpe in every split, not a silent NaN."""
    constant = np.zeros((10, 400))
    with pytest.raises(ValueError, match='no usable splits'):
        probability_of_backtest_overfitting(constant, n_blocks=8)


def test_pbo_verdict_bands():
    def make(pbo):
        return PboResult(pbo=pbo, logit=0.0, n_splits=1, n_configurations=1,
                         is_sqrt=0.0, median_is_sharpe=0.0, performance_degradation=0.0)

    assert make(0.1).verdict == 'UNLIKELY_TO_BE_OVERFIT'
    assert make(0.3).verdict == 'AMBIGUOUS'
    assert make(0.6).verdict == 'LIKELY_OVERFIT'
    assert make(0.9).verdict == 'HIGHLY_LIKELY_OVERFIT'
    assert make(0.6).concerning
    assert not make(0.3).concerning


# --- deflated sharpe ------------------------------------------------------


def test_more_trials_means_a_higher_selection_penalty():
    values = np.random.default_rng(2).normal(0.002, 0.01, 800)
    few = deflated_sharpe(values, n_trials=1)
    many = deflated_sharpe(values, n_trials=1000)

    assert many.deflated_sharpe < few.deflated_sharpe
    assert many.expected_max_sharpe > few.expected_max_sharpe


def test_a_real_edge_survives_deflation_and_noise_does_not():
    real = np.random.default_rng(3).normal(0.005, 0.01, 900)
    noise = np.random.default_rng(4).normal(0.0, 0.01, 900)

    assert deflated_sharpe(real, n_trials=50).verdict == 'SURVIVES_DEFLATION'
    assert deflated_sharpe(noise, n_trials=50).verdict == 'NO_EDGE_AFTER_DEFLATION'


def test_deflation_can_never_raise_a_sharpe():
    values = np.random.default_rng(5).normal(0.001, 0.01, 600)
    result = deflated_sharpe(values, n_trials=10)
    assert result.deflated_sharpe <= result.observed_sharpe


def test_trial_sharpes_improve_the_variance_estimate():
    values = np.random.default_rng(6).normal(0.004, 0.01, 700)
    without = deflated_sharpe(values, n_trials=20)
    with_trials = deflated_sharpe(
        values, n_trials=20, trial_sharpes=[2.0, 0.5, 1.2, 0.1, 3.0]
    )
    assert with_trials.expected_max_sharpe != without.expected_max_sharpe
    assert with_trials.n_trials == 5


def test_deflated_sharpe_validates_its_inputs():
    values = np.random.default_rng(7).normal(0.001, 0.01, 200)
    with pytest.raises(ValueError):
        deflated_sharpe(np.array([1.0]), n_trials=5)
    with pytest.raises(ValueError):
        deflated_sharpe(values, n_trials=0)
    with pytest.raises(ValueError, match='not finite'):
        deflated_sharpe(np.full(100, 0.01), n_trials=5)


def test_expected_max_sharpe_grows_with_trial_count():
    assert expected_max_sharpe(1000) > expected_max_sharpe(10)
    # A single trial cannot have been selected, so there is nothing to deflate.
    assert expected_max_sharpe(1) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(0)


def test_deflated_result_significance_is_exposed():
    result = deflated_sharpe(np.random.default_rng(8).normal(0.006, 0.01, 900), n_trials=20)
    assert isinstance(result, DeflatedSharpeResult)
    assert result.significant is True
    assert 'deflated_sharpe' in result.as_dict()


# --- plateau --------------------------------------------------------------


def test_a_broad_plateau_is_recognised():
    grid = np.concatenate([np.full(30, 2.0), np.full(10, 0.1)])
    result = plateau_analysis(grid)
    assert result.is_plateau
    assert result.verdict == 'BROAD_PLATEAU'
    assert result.within_10pct == 30


def test_a_narrow_spike_is_flagged():
    grid = np.full(50, 0.1)
    grid[7] = 3.0
    result = plateau_analysis(grid)
    assert not result.is_plateau
    assert result.verdict == 'SPIKE_FITTED'
    assert result.within_10pct == 1


def test_plateau_analysis_rejects_bad_grids():
    with pytest.raises(ValueError):
        plateau_analysis(np.array([]))
    with pytest.raises(ValueError):
        plateau_analysis(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        plateau_analysis(np.ones(10), min_fraction=0.0)
    with pytest.raises(ValueError):
        plateau_analysis(np.ones(10), min_relative_width=-1.0)


def test_plateau_keeps_the_axis_description():
    grid = np.ones(10)
    grid[3] = 5.0
    result = plateau_analysis(grid, axes={'fast': [5, 20, 50]})
    assert isinstance(result, PlateauResult)
    assert result.grid['fast'] == [5, 20, 50]


# --- the combined report --------------------------------------------------


def test_a_fitted_strategy_fails_the_combined_report():
    rng = np.random.default_rng(9)
    matrix = rng.normal(0.0, 0.01, (60, 800))
    # One configuration was picked for its best-looking result: pure selection.
    best = matrix[int(np.argmax([sharpe_ratio(r) for r in matrix]))]

    report = overfitting_report(matrix, best, n_blocks=8)
    assert report['verdict'].startswith('FAILS')
    assert 'pbo' in report and 'deflated_sharpe' in report and 'plateau' in report


def test_a_real_edge_passes_the_combined_report():
    # Several configurations share a genuine edge of similar strength, which
    # is what a real effect looks like once noise is smoothed over.
    rng = np.random.default_rng(21)
    alphas = np.linspace(0.006, 0.001, 12)
    matrix = rng.normal(0.0, 0.01, (12, 1200)) + alphas[:, None]
    report = overfitting_report(matrix, matrix[0], n_blocks=8)

    assert report['pbo']['pbo'] < 0.5
    assert report['plateau']['is_plateau'] is True
    assert report['deflated_sharpe']['deflated_sharpe'] > 0
    assert report['verdict'] == 'PASSES_OVERFITTING_CHECKS'


def test_a_single_lucky_winner_among_noise_is_a_spike_not_a_plateau():
    """One good configuration out of pure noise is exactly what fitting looks like."""
    matrix = trend_matrix(seed=21, n_configs=12, n_bars=1200)
    report = overfitting_report(matrix, matrix[0], n_blocks=8)

    assert report['plateau']['is_plateau'] is False
    assert report['plateau']['verdict'] == 'SPIKE_FITTED'


def test_the_report_explains_every_failure():
    rng = np.random.default_rng(13)
    matrix = rng.normal(0.0, 0.01, (40, 900))
    report = overfitting_report(matrix, matrix[0], n_blocks=8)
    assert report['verdict'].startswith('FAILS: ')
    assert len(report['verdict']) > len('FAILS: ')


def test_the_report_requires_more_than_one_configuration():
    with pytest.raises(ValueError, match='at least 2 configurations'):
        overfitting_report(np.zeros((1, 100)), np.zeros(100))
