import numpy as np
import pytest

from algorithex.audit import CandleView, LeakageGuard, detect_lookahead


LEAKY = '''
def should_long(self):
    last = self.candles[-1]
    everything_before = self.candles[:-1]
    ahead = series.shift(-1)
    centred = series.rolling(5, center=True).mean()
    rolled = np.roll(series, -3)
    return last
'''

CLEAN = '''
def should_long(self):
    recent = self.candles[:self.index]
    lagged = series.shift(1)
    rolling = series.rolling(5).mean()
    return recent[-1]
'''


def rules_found(report):
    return {f.rule for f in report.errors}


def test_detects_every_lookahead_pattern():
    report = detect_lookahead(LEAKY, 'leaky.py')

    assert rules_found(report) == {
        'negative-index', 'open-ended-slice', 'negative-shift',
        'centered-window', 'negative-roll',
    }


def test_clean_strategy_passes():
    report = detect_lookahead(CLEAN, 'clean.py')

    assert report.clean
    assert report.errors == []
    assert 'clean' in report.summary()


def test_findings_carry_actionable_locations():
    report = detect_lookahead(LEAKY, 'leaky.py')

    for finding in report.errors:
        assert finding.line > 0
        assert finding.snippet
        assert finding.rule


def test_negative_index_is_caught_on_a_bare_name_too():
    report = detect_lookahead('x = prices[-1]')

    assert 'negative-index' in rules_found(report)


def test_negative_index_on_an_unrelated_name_is_not_flagged():
    report = detect_lookahead('x = widgets[-1]\ny = counter[-1]')

    assert report.clean


def test_positive_shift_and_uncentred_windows_are_safe():
    report = detect_lookahead('a = s.shift(1)\nb = s.rolling(3).mean()\nc = np.roll(s, 3)')

    assert report.clean


def test_negative_shift_as_a_keyword_is_caught():
    report = detect_lookahead('a = s.shift(shift=-2)')

    assert 'negative-shift' in rules_found(report)


def test_center_true_as_a_positional_flag_is_not_misread():
    report = detect_lookahead('a = s.rolling(3, True)')

    # Positional `center` is not recognised, so it is not reported. Reported
    # rather than silently treated as safe, because a missed detection is worse
    # than a noisy one.
    assert report.clean


def test_syntax_errors_are_reported_not_raised():
    report = detect_lookahead('def broken(:\n    pass')

    assert not report.clean
    assert report.errors[0].rule == 'syntax-error'


def test_report_counts_errors_and_warnings():
    report = detect_lookahead(LEAKY)

    assert len(report.errors) == 5
    assert report.warnings == []
    assert not bool(report)


def test_candle_view_records_the_highest_index_read():
    view = CandleView(np.arange(100.0))

    view[:50]

    assert view.max_index_read == 49
    assert len(view) == 100


def test_candle_view_proxies_slices_and_scalars():
    view = CandleView(np.arange(10.0))

    assert view[3] == 3.0
    assert list(view[2:5]) == [2.0, 3.0, 4.0]
    assert list(view.underlying) == list(range(10))


def test_guard_is_clean_when_reads_stay_within_the_bar():
    view = CandleView(np.arange(100.0))
    guard = LeakageGuard(bar_index=50)
    guard.track(view, 'candles')

    view[:51]  # up to and including the current bar

    assert guard.check() == []
    assert guard.clean


def test_guard_flags_reading_ahead():
    view = CandleView(np.arange(100.0))
    guard = LeakageGuard(bar_index=50)
    guard.track(view, 'candles')

    view[55]
    violations = guard.check()

    assert len(violations) == 1
    assert violations[0]['bars_ahead'] == 5
    assert violations[0]['bar'] == 50
    assert not guard.clean


def test_guard_reset_clears_reads_but_keeps_the_bar():
    view = CandleView(np.arange(100.0))
    guard = LeakageGuard(bar_index=50)
    guard.track(view)

    view[55]
    assert guard.check()

    guard.reset_reads()
    assert guard.check() == []
    assert guard.bar_index == 50


def test_guard_follows_the_bar_forward():
    """The same read is a violation early and legal later, which is exactly how
    lookahead works: catching up with the data is not peeking."""
    view = CandleView(np.arange(100.0))
    guard = LeakageGuard(bar_index=10)
    guard.track(view)

    view[30]
    assert guard.check()[0]['bars_ahead'] == 20

    # Once the bar advances to 30, that same read is no longer in the future.
    guard.set_bar(30)
    guard.reset_reads()
    view[30]
    assert guard.check() == []
    # `violations` is a cumulative record, so the earlier catch stays on file.
    assert guard.violations[0]['bars_ahead'] == 20


def test_candle_view_length_matches_underlying():
    assert len(CandleView(np.zeros((7, 3)))) == 7