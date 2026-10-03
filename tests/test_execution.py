import numpy as np
import pytest

from algorithex.execution import (
    BookLevel,
    CompositeSlippage,
    FixedRateSlippage,
    OrderBook,
    SpreadSlippage,
    SquareRootImpact,
    VolatilitySlippage,
    assess_parity,
    compare_returns,
    cost_of_order,
    fill_deviation,
    funding_cost,
    simulate_fill,
)


def test_slippage_always_hurts_the_trader():
    models = [
        FixedRateSlippage(10), SpreadSlippage(20), VolatilitySlippage(0.5), SquareRootImpact(0.2),
    ]
    for model in models:
        buy = cost_of_order('buy', 100.0, 5, slippage=model, volatility=0.02, volume=500)
        sell = cost_of_order('sell', 100.0, 5, slippage=model, volatility=0.02, volume=500)
        assert buy.fill_price > 100.0, f'{type(model).__name__} helped a buy'
        assert sell.fill_price < 100.0, f'{type(model).__name__} helped a sell'


def test_spread_slippage_is_half_the_spread():
    model = SpreadSlippage(20)

    assert model.penalty('buy', 1.0) == pytest.approx(0.001, rel=1e-6)


def test_passive_order_pays_no_spread():
    assert SpreadSlippage(20, marketable=False).penalty('buy', 1.0) == 0.0


def test_square_root_impact_is_sublinear_in_size():
    model = SquareRootImpact(0.1, 0.05)

    small = model.penalty('buy', 1.0, 0.02, 10_000)
    big = model.penalty('buy', 100.0, 0.02, 10_000)

    # A hundred times the size costs ten times the penalty, not a hundred.
    assert big / small == pytest.approx(10.0, rel=1e-6)


def test_square_root_impact_is_zero_without_a_size():
    assert SquareRootImpact().penalty('buy', 0.0, 0.02, 1000) == 0.0


def test_composite_slippage_sums_its_parts():
    composite = CompositeSlippage((FixedRateSlippage(10), FixedRateSlippage(20)))

    assert composite.penalty('buy', 1.0) == pytest.approx(0.003)


def test_composite_slippage_requires_a_model():
    with pytest.raises(ValueError):
        CompositeSlippage(())


def test_volatility_slippage_has_a_floor():
    model = VolatilitySlippage(0.1, floor=0.01)

    # A dead-flat stretch still costs something rather than becoming free.
    assert model.penalty('buy', 1.0, volatility=0.0) == pytest.approx(0.001)


def test_cost_itemises_into_slippage_and_impact():
    cost = cost_of_order(
        'buy', 100.0, 10,
        slippage=FixedRateSlippage(10), impact=SquareRootImpact(0.1),
        commission_rate=0.001, volatility=0.02, volume=1000,
    )

    assert cost.slippage_cost > 0
    assert cost.impact_cost > 0
    assert cost.commission == pytest.approx(1.0)
    assert cost.total_cost == pytest.approx(cost.slippage_cost + cost.impact_cost + cost.commission)
    assert cost.total_bps > 0


def test_zero_cost_when_no_models_are_supplied():
    cost = cost_of_order('buy', 100.0, 10)

    assert cost.fill_price == pytest.approx(100.0)
    assert cost.total_cost == 0.0


def test_cost_rejects_invalid_orders():
    with pytest.raises(ValueError):
        cost_of_order('buy', 0.0, 1)
    with pytest.raises(ValueError):
        cost_of_order('buy', 100.0, -1)
    with pytest.raises(ValueError):
        FixedRateSlippage(-1).penalty('buy', 1.0)


def test_funding_pays_longs_and_pays_shorts_when_negative():
    assert funding_cost(0.0001, 10_000, 3, 'long') > 0
    assert funding_cost(0.0001, 10_000, 3, 'short') < 0
    # Negative funding reverses who pays.
    assert funding_cost(-0.0001, 10_000, 3, 'long') < 0


def test_funding_rejects_negative_intervals():
    with pytest.raises(ValueError):
        funding_cost(0.0001, 1000, -1)


def test_order_book_reports_mid_and_spread():
    book = OrderBook(
        bids=[BookLevel(99.5, 50), BookLevel(99.0, 100)],
        asks=[BookLevel(100.5, 30), BookLevel(101.0, 200)],
    )

    assert book.mid == pytest.approx(100.0)
    assert book.spread == pytest.approx(1.0)
    assert book.is_crossed() is False


def test_small_order_fills_at_the_touch_only():
    book = OrderBook(bids=[BookLevel(99.5, 50)], asks=[BookLevel(100.5, 30)])

    fill = simulate_fill(book, 'buy', 20)

    assert fill.fully_filled
    assert fill.levels_consumed == 1
    assert fill.volume_weighted_price == pytest.approx(100.5)


def test_large_order_walks_the_ladder_at_a_worse_average_price():
    book = OrderBook(bids=[BookLevel(99.5, 50)], asks=[BookLevel(100.5, 30), BookLevel(101.0, 200)])

    small = simulate_fill(book, 'buy', 20)
    large = simulate_fill(book, 'buy', 100)

    assert large.levels_consumed == 2
    assert large.volume_weighted_price > small.volume_weighted_price
    # Volume weighted, so between the two levels touched.
    expected = (30 * 100.5 + 70 * 101.0) / 100
    assert large.volume_weighted_price == pytest.approx(expected)


def test_sell_consumes_bids_not_asks():
    book = OrderBook(bids=[BookLevel(99.5, 40)], asks=[BookLevel(100.5, 40)])

    fill = simulate_fill(book, 'sell', 10)

    assert fill.volume_weighted_price == pytest.approx(99.5)


def test_order_larger_than_the_book_partially_fills():
    book = OrderBook(bids=[BookLevel(99.5, 10)], asks=[BookLevel(100.5, 10)])

    fill = simulate_fill(book, 'buy', 100)

    assert fill.filled == 10.0
    assert fill.fill_ratio == pytest.approx(0.1)
    assert not fill.fully_filled


def test_queue_position_blocks_a_passive_order():
    book = OrderBook(bids=[BookLevel(99.5, 50)], asks=[BookLevel(100.5, 50)])

    blocked = simulate_fill(book, 'sell', 40, queue_ahead=45, marketable=False)
    partial = simulate_fill(book, 'sell', 40, queue_ahead=45 - 30, marketable=False)

    assert blocked.filled == pytest.approx(5.0)
    assert partial.filled == pytest.approx(35.0)


def test_passive_order_with_a_full_queue_does_not_fill():
    book = OrderBook(bids=[BookLevel(99.5, 50)], asks=[BookLevel(100.5, 50)])

    fill = simulate_fill(book, 'sell', 10, queue_ahead=50, marketable=False)

    assert fill.filled == 0.0


def test_empty_book_fills_nothing():
    fill = simulate_fill(OrderBook(), 'buy', 10)

    assert fill.filled == 0.0
    assert fill.volume_weighted_price is None


def test_book_level_and_fill_reject_bad_input():
    with pytest.raises(ValueError):
        BookLevel(0.0, 10)
    with pytest.raises(ValueError):
        BookLevel(100.0, -1)
    with pytest.raises(ValueError):
        simulate_fill(OrderBook(), 'buy', -1)


def test_fill_deviation_sign_depends_on_side():
    # Same price move, opposite consequence: a buy that fills higher paid more
    # (worse), while a sell that fills higher received more (better).
    deviation = fill_deviation([100.0, 100.0], [101.0, 101.0], sides=['buy', 'sell'])

    assert deviation[0] > 0
    assert deviation[1] < 0
    assert deviation[0] == pytest.approx(-deviation[1])


def test_fill_deviation_is_negative_when_live_beats_the_backtest():
    deviation = fill_deviation([100.0], [99.0], sides=['buy'])

    assert deviation[0] < 0


def test_parity_flags_a_systematically_optimistic_backtest():
    rng = np.random.default_rng(0)
    assumed = np.full(300, 100.0)
    # Fills consistently 0.5% worse than assumed, which is 50 bps.
    realized = assumed * (1.0 + rng.normal(0.005, 0.0005, 300))

    report = assess_parity(assumed, realized, sides=['buy'] * 300)

    assert report.systematic is True
    assert report.mean_signed_bps == pytest.approx(50, rel=0.1)
    assert report.bias_ratio > 0.9
    assert 'optimistic' in report.verdict


def test_parity_tolerates_a_small_bias():
    rng = np.random.default_rng(21)
    assumed = np.full(200, 100.0)
    realized = assumed * (1.0 + rng.normal(0.0001, 0.00005, 200))

    report = assess_parity(assumed, realized, sides=['buy'] * 200)

    assert report.systematic is False


def test_parity_does_not_flag_symmetric_noise():
    rng = np.random.default_rng(1)
    assumed = np.full(300, 100.0)
    realized = assumed + rng.normal(0.0, 0.002, 300)

    report = assess_parity(assumed, realized, sides=['buy'] * 300)

    assert report.systematic is False
    assert report.bias_ratio < 0.3


def test_parity_breaks_down_by_symbol():
    assumed = [100.0] * 4
    realized = [101.0, 101.0, 100.5, 100.5]

    report = assess_parity(assumed, realized, sides=['buy'] * 4, symbols=['BTC', 'ETH', 'BTC', 'ETH'])

    assert set(report.per_symbol) == {'BTC', 'ETH'}
    assert report.per_symbol['BTC']['observations'] == 2


def test_parity_handles_no_fills():
    report = assess_parity([], [])

    assert report.observations == 0
    assert 'no fills' in report.verdict


def test_parity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fill_deviation([100.0, 101.0], [100.0])
    with pytest.raises(ValueError):
        fill_deviation([0.0, 1.0], [1.0, 1.0])


def test_compare_returns_reports_gaps():
    rng = np.random.default_rng(2)
    backtest = rng.normal(0.001, 0.01, 400)
    # Same signal, more noise: Sharpe must fall even though the mean is intact.
    live = 0.001 + (backtest - 0.001) * 2.0 + rng.normal(0, 0.004, 400)

    comparison = compare_returns(backtest, live)

    assert comparison['backtest']['sharpe'] > comparison['live']['sharpe']
    assert comparison['volatility_ratio'] > 1.0
    assert 0.0 <= comparison['correlation'] <= 1.0


def test_compare_returns_is_invariant_to_uniform_scaling():
    """Sharpe is scale-free, so shrinking every return cannot change it. A
    comparison that reports a Sharpe gap here would be reporting a fake
    discrepancy."""
    rng = np.random.default_rng(21)
    backtest = rng.normal(0.001, 0.01, 400)

    comparison = compare_returns(backtest, backtest * 0.5)

    assert comparison['sharpe_gap'] == pytest.approx(0.0, abs=1e-9)
    assert comparison['volatility_ratio'] == pytest.approx(0.5, rel=1e-9)


def test_compare_returns_requires_aligned_series():
    with pytest.raises(ValueError):
        compare_returns(np.zeros(10), np.zeros(9))
    with pytest.raises(ValueError):
        compare_returns(np.zeros(1), np.zeros(1))