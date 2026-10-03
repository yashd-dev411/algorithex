import numpy as np
import pytest

from algorithex.research.certification import (
    FAIL,
    GRADES,
    NOT_RUN,
    PASS,
    WARN,
    CertificationReport,
    Check,
    certify,
    grade_for,
    render_report,
)

CLEAN_SOURCE = """
def go(self):
    price = self.price
    if self.close > price:
        return 1
    return 0
"""

CHEATING_SOURCE = """
def go(self):
    future = self.candles[-1]['close']
    past = self.candles[-2]['close']
    return 1 if future > past else 0
"""


def market(seed=4, n=800, drift=0.0009, vol=0.012):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def hold(prices):
    return float(prices[-1] - prices[0])


def momentum(prices, window=20):
    if prices.size <= window + 1:
        return 0.0
    position = np.sign(prices[window:-1] - prices[:-window - 1])
    return float(np.sum(position * (prices[window + 1:] / prices[window:-1] - 1.0)))


def loser(prices):
    return -1.0


def broad_configurations(seed=31, n_configs=12, n_bars=900):
    """Several configurations sharing a genuine edge: not a spike."""
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.006, 0.0015, n_configs)
    return rng.normal(0.0, 0.01, (n_configs, n_bars)) + alphas[:, None]


def noisy_configurations(seed=32, n_configs=40, n_bars=900):
    return np.random.default_rng(seed).normal(0.0, 0.01, (n_configs, n_bars))


def good_windows(n=6, sharpe=1.4):
    return [{'sharpe': sharpe - 0.05 * i} for i in range(n)]


def mixed_windows(n=6):
    return [{'sharpe': s} for s in (2.0, 1.1, 0.4, -0.3, -1.1, -2.4)]


# --- grading --------------------------------------------------------------


def test_grades_map_onto_the_expected_bands():
    assert grade_for(0.95) == 'A'
    assert grade_for(0.75) == 'B'
    assert grade_for(0.60) == 'C'
    assert grade_for(0.45) == 'D'
    assert grade_for(0.10) == 'F'
    assert all(grade_for(s) in GRADES for s in np.linspace(0, 1, 21))


def test_a_blocking_failure_caps_the_grade_at_f():
    assert grade_for(0.99, has_blocking_failure=True) == 'F'


def test_grading_clamps_out_of_range_scores():
    assert grade_for(5.0) == 'A'
    assert grade_for(-3.0) == 'F'
    assert grade_for(float('nan')) == 'F'


# --- check objects --------------------------------------------------------


def test_a_check_validates_its_own_fields():
    with pytest.raises(ValueError, match='status must be'):
        Check('x', 'excellent', 1.0, 'detail')
    with pytest.raises(ValueError, match='score must be'):
        Check('x', PASS, 1.5, 'detail')
    with pytest.raises(ValueError, match='weight must be'):
        Check('x', PASS, 0.5, 'detail', weight=0.0)
    assert Check('x', PASS, 0.5, 'detail').passed


# --- individual checks ----------------------------------------------------


def test_a_clean_strategy_passes_the_lookahead_check():
    report = certify(source_code=CLEAN_SOURCE)
    assert report.check('lookahead').passed
    assert report.check('lookahead').blocking


def test_a_cheating_strategy_fails_the_lookahead_check_blockingly():
    report = certify(source_code=CHEATING_SOURCE)
    check = report.check('lookahead')

    assert check.status == FAIL
    assert check.blocking
    assert report.blocking_failures
    assert report.verdict == 'REJECTED'
    assert report.grade == 'F'


def test_a_source_file_can_be_audited_by_path(tmp_path):
    path = tmp_path / 'strategy.py'
    path.write_text(CLEAN_SOURCE, encoding='utf-8')
    report = certify(source_path=str(path))
    assert report.check('lookahead').passed


def test_an_unreadable_source_is_a_failure_not_a_crash(tmp_path):
    report = certify(source_path=str(tmp_path / 'does_not_exist.py'))
    assert report.check('lookahead').status == FAIL
    assert report.blocking_failures


def test_overfitting_is_scored_from_the_probability_not_the_verdict():
    strong = certify(
        configuration_returns=broad_configurations(),
        best_returns=broad_configurations()[0],
    )
    weak = certify(
        configuration_returns=noisy_configurations(),
        best_returns=noisy_configurations()[0],
    )
    assert strong.check('overfitting').score > weak.check('overfitting').score
    assert 'PBO' in strong.check('overfitting').detail


def test_a_cost_breaker_is_caught_by_the_cost_check():
    prices = market()
    generous = certify(strategy=hold, prices=prices, required_cost_multiple=0.01)
    harsh = certify(strategy=hold, prices=prices, required_cost_multiple=1000.0)

    assert generous.check('cost_survival').passed
    assert not harsh.check('cost_survival').passed


def test_a_losing_strategy_fails_robustness_blockingly():
    report = certify(strategy=loser, prices=market())
    check = report.check('robustness')

    assert check.status == FAIL
    assert check.blocking
    assert report.verdict == 'REJECTED'


def test_a_hold_strategy_survives_the_robustness_attacks():
    report = certify(strategy=hold, prices=market())
    assert report.check('robustness').score > 0


def test_out_of_sample_scores_the_worst_window_not_the_mean():
    consistent = certify(window_results=good_windows())
    lumpy = certify(window_results=mixed_windows())

    assert consistent.check('out_of_sample').passed
    assert not lumpy.check('out_of_sample').passed
    assert 'worst' in lumpy.check('out_of_sample').detail


# --- accounting -----------------------------------------------------------


def test_a_check_with_no_input_is_not_run_and_costs_you_the_grade():
    report = certify(strategy_name='bare')
    assert len(report.not_run) == 5
    assert report.coverage == 0.0
    assert report.grade == 'F'
    assert report.score == 0.0
    assert not report.deployable


def test_dropping_a_check_cannot_improve_the_score():
    """Otherwise the incentive is to test less."""
    partial = certify(strategy=hold, prices=market())
    complete = certify(
        strategy=hold,
        prices=market(),
        configuration_returns=broad_configurations(),
        best_returns=broad_configurations()[0],
        window_results=good_windows(),
    )
    assert complete.score > partial.score
    assert complete.coverage > partial.coverage


def test_a_full_report_of_a_good_strategy_certifies():
    matrix = broad_configurations()
    report = certify(
        strategy_name='broad trend',
        source_code=CLEAN_SOURCE,
        configuration_returns=matrix,
        best_returns=matrix[0],
        strategy=hold,
        prices=market(),
        window_results=good_windows(),
    )

    assert report.coverage == 1.0
    assert report.grade in ('A', 'B')
    assert report.verdict == 'CERTIFIED'
    assert report.deployable
    assert report.reasons() == []


def test_a_high_turnover_strategy_is_rejected_by_the_cost_check():
    """20-bar momentum across 800 bars pays far more in costs than it makes."""
    report = certify(
        strategy=momentum,
        prices=market(drift=0.0002),
        required_cost_multiple=3.0,
    )
    check = report.check('cost_survival')
    assert check.status in (FAIL, WARN)
    assert not report.deployable


def test_coverage_is_the_weighted_fraction_that_actually_ran():
    matrix = broad_configurations()
    report = certify(
        strategy=hold,
        prices=market(),
        configuration_returns=matrix,
        best_returns=matrix[0],
    )
    # Weights are 1.5 each for lookahead/overfitting/out-of-sample and 1.0 each
    # for robustness/costs, so 6.5 in total; 3.0 of that weight did not run.
    assert report.coverage == pytest.approx(3.5 / 6.5)
    assert [c.name for c in report.not_run] == ['lookahead', 'out_of_sample']


def test_an_empty_report_scores_zero():
    report = CertificationReport(strategy_name='empty', checks=[])
    assert report.score == 0.0
    assert report.coverage == 0.0
    assert report.grade == 'F'


def test_looking_up_an_unknown_check_is_an_error_with_the_options():
    report = certify(strategy=hold, prices=market())
    with pytest.raises(KeyError, match='available'):
        report.check('vibes')


def test_a_missing_input_names_what_is_actually_absent():
    only_prices = certify(prices=market())
    assert 'no executable strategy' in only_prices.check('robustness').detail
    assert 'no executable strategy' in only_prices.check('cost_survival').detail

    only_strategy = certify(strategy=hold)
    assert 'supply a price series' in only_strategy.check('robustness').detail

    neither = certify()
    assert 'and a price series' in neither.check('robustness').detail


# --- reporting ------------------------------------------------------------


def test_reasons_are_ordered_worst_first():
    report = certify(source_code=CHEATING_SOURCE, window_results=[])
    lines = report.reasons()
    assert lines[0].startswith('BLOCKING')
    assert any(line.startswith('NOT RUN') for line in lines)


def test_the_report_serialises_and_renders():
    report = certify(
        strategy_name='mixed',
        source_code=CLEAN_SOURCE,
        strategy=hold,
        prices=market(),
        window_results=mixed_windows(),
    )
    payload = report.as_dict()
    assert payload['grade'] in GRADES
    assert len(payload['checks']) == 5

    text = render_report(report)
    assert 'mixed' in text
    assert 'Grade:' in text
    assert text.count('\n') > 10
    assert report.render() == text


def test_a_clean_render_says_no_findings():
    matrix = broad_configurations()
    report = certify(
        source_code=CLEAN_SOURCE,
        configuration_returns=matrix,
        best_returns=matrix[0],
        strategy=hold,
        prices=market(),
        window_results=good_windows(),
    )
    assert 'No findings.' in render_report(report)
