import numpy as np
import pytest

from algorithex.research.attribution import (
    TradeRecord,
    attribute_trades,
    attribution_report,
    execution_quality,
    ridge_attribution,
    trade_pnl,
    univariate_attribution,
)


def sample_trades():
    return [
        {'symbol': 'BTC', 'side': 'long', 'entry_price': 100, 'exit_price': 110, 'quantity': 2, 'fees': 0.5},
        {'symbol': 'BTC', 'side': 'short', 'entry_price': 110, 'exit_price': 105, 'quantity': 1, 'fees': 0.5},
        {'symbol': 'ETH', 'side': 'long', 'entry_price': 50, 'exit_price': 47, 'quantity': 3, 'fees': 0.5},
    ]


def test_long_pnl_is_price_move_times_quantity_less_fees():
    trade = {'symbol': 'BTC', 'side': 'long', 'entry_price': 100, 'exit_price': 110, 'quantity': 2, 'fees': 0.5}

    assert trade_pnl(trade) == pytest.approx(19.5)


def test_short_pnl_gains_when_price_falls():
    trade = {'symbol': 'BTC', 'side': 'short', 'entry_price': 110, 'exit_price': 105, 'quantity': 1}

    assert trade_pnl(trade) == pytest.approx(5.0)


def test_short_pnl_loses_when_price_rises():
    trade = {'symbol': 'BTC', 'side': 'short', 'entry_price': 100, 'exit_price': 110, 'quantity': 1}

    assert trade_pnl(trade) == pytest.approx(-10.0)


def test_trade_record_computes_the_same_thing_as_a_mapping():
    record = TradeRecord('BTC', 'long', 100, 110, 2, 0.5)

    assert trade_pnl(record) == pytest.approx(trade_pnl({
        'symbol': 'BTC', 'side': 'long', 'entry_price': 100, 'exit_price': 110, 'quantity': 2, 'fees': 0.5,
    }))


def test_attribute_trades_totals_add_up():
    report = attribute_trades(sample_trades())

    assert report['trades'] == 3
    assert report['net_pnl'] == pytest.approx(19.5 + 4.5 - 9.5)
    assert report['gross_pnl'] - report['fees'] == pytest.approx(report['net_pnl'])
    assert report['win_rate'] == pytest.approx(2 / 3)


def test_attribute_trades_breaks_down_by_symbol_and_side():
    report = attribute_trades(sample_trades())

    assert report['by_symbol']['BTC']['net_pnl'] == pytest.approx(24.0)
    assert report['by_symbol']['ETH']['net_pnl'] == pytest.approx(-9.5)
    assert report['by_side']['long']['net_pnl'] == pytest.approx(10.0)
    assert report['by_side']['short']['net_pnl'] == pytest.approx(4.5)


def test_attribute_trades_flags_single_trade_dependence():
    report = attribute_trades(sample_trades())

    # One trade out of two winners carries most of the profit.
    assert report['profit_concentration'] > 0.5


def test_attribute_trades_accepts_a_time_bucket():
    trades = [dict(t, opened_at=0, closed_at=i * 1000) for i, t in enumerate(sample_trades())]

    report = attribute_trades(trades, time_bucket=lambda ts: f'p{ts // 2000}')

    assert set(report['by_period']) == {'p0', 'p1'}


def test_attribute_trades_needs_at_least_one_trade():
    with pytest.raises(ValueError):
        attribute_trades([])


def test_execution_quality_reports_capture_ratio():
    trades = [
        {'symbol': 'X', 'side': 'long', 'entry_price': 100, 'exit_price': 108, 'quantity': 1, 'mfe': 12, 'mae': -3},
        {'symbol': 'X', 'side': 'long', 'entry_price': 100, 'exit_price': 95, 'quantity': 1, 'mfe': 9, 'mae': -6},
    ]

    quality = execution_quality(trades)

    assert quality['available'] is True
    assert quality['measured_trades'] == 2
    assert 0.0 < quality['mean_capture_ratio'] < 1.0
    assert quality['losing_share'] == pytest.approx(0.5)


def test_execution_quality_says_so_when_excursions_are_missing():
    quality = execution_quality(sample_trades())

    # Reporting a fabricated zero capture rate would be worse than admitting
    # the data was never supplied.
    assert quality['available'] is False
    assert 'reason' in quality


def signal_features(seed=5, n=400):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    features = {
        'real_signal': signal,
        'noise_a': rng.normal(size=n),
        'duplicate': signal * rng.normal(1.0, 0.1, n),
    }
    outcomes = 3.0 * signal + rng.normal(0, 0.5, n)
    return features, outcomes


def test_univariate_attribution_ranks_the_real_signal_first():
    features, outcomes = signal_features()

    scores = univariate_attribution(features, outcomes)

    assert scores['real_signal']['rank'] == 1
    assert abs(scores['real_signal']['correlation']) > abs(scores['noise_a']['correlation'])


def test_univariate_attribution_detects_the_correct_sign():
    features, outcomes = signal_features()

    scores = univariate_attribution(features, outcomes)

    assert scores['real_signal']['correlation'] > 0.9


def test_ridge_attribution_credits_the_signal_and_ignores_noise():
    features, outcomes = signal_features()

    contributions = ridge_attribution(features, outcomes, alpha=0.1)

    assert abs(contributions['real_signal']) > abs(contributions['noise_a'])


def test_ridge_attribution_penalty_is_scale_invariant():
    """Standardising before the penalty means a feature's units cannot buy it
    extra regularisation. The reported coefficient rescales inversely with the
    feature, so the invariant to check is the contribution it makes to a fit."""
    features, outcomes = signal_features()
    scaled = {name: values * 1000.0 for name, values in features.items()}

    base = ridge_attribution(features, outcomes)
    rescaled = ridge_attribution(scaled, outcomes)

    # Coefficient times feature value must be identical either way.
    for name in features:
        assert np.isclose(
            base[name] * features['real_signal'][0],
            rescaled[name] * scaled['real_signal'][0],
            rtol=1e-6,
        )


def test_ridge_attribution_credits_scale_invariant_features_equally():
    rng = np.random.default_rng(9)
    signal = rng.normal(size=300)
    # Two features carrying identical information, expressed in very different units.
    features = {'small_units': signal, 'huge_units': signal * 1e6}
    outcomes = 2.0 * signal + rng.normal(0, 0.5, 300)

    coefficients = ridge_attribution(features, outcomes, alpha=1.0)

    # Neither unit scale may win extra credit for the same information, so the
    # two must make identical contributions to the fitted value at every point.
    for index in range(signal.size):
        assert np.isclose(
            coefficients['small_units'] * features['small_units'][index],
            coefficients['huge_units'] * features['huge_units'][index],
            rtol=1e-6,
        )


def test_attribution_report_summarises_both_views():
    features, outcomes = signal_features()

    report = attribution_report(features, outcomes)

    assert report['features'] == 3
    assert report['observations'] == 400
    assert report['dominant_features'][0] == 'real_signal'
    assert report['r_squared_in_sample'] > 0.9


def test_attribution_ignores_non_finite_observations():
    features, outcomes = signal_features(n=100)
    features = dict(features)
    features['real_signal'] = features['real_signal'].copy()
    features['real_signal'][0] = np.nan

    report = attribution_report(features, outcomes)

    assert report['observations'] == 99


def test_attribution_rejects_mismatched_shapes():
    features, outcomes = signal_features(n=50)

    with pytest.raises(ValueError):
        univariate_attribution(features, outcomes[:10])
    with pytest.raises(ValueError):
        univariate_attribution({}, outcomes)
    with pytest.raises(ValueError):
        ridge_attribution({'a': np.zeros(50)}, np.full(50, np.nan))