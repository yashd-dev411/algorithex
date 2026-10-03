import numpy as np
import pytest

from algorithex.research.walkforward import (
    anchored_windows,
    combinatorial_paths,
    combinatorial_purged_split,
    evaluate_windows,
    performance_metrics,
    purge_train_indices,
    purged_kfold,
    purged_walk_forward,
    rolling_windows,
    stitch_oos_returns,
)


def test_rolling_windows_keep_training_length_constant():
    windows = rolling_windows(100, 40, 20)

    assert len(windows) == 3
    assert all(w.train_size == 40 for w in windows)
    # Training slides forward rather than growing.
    assert windows[0].train[0] == 0
    assert windows[1].train[0] == 20


def test_anchored_windows_expand_from_the_first_bar():
    windows = anchored_windows(100, 40, 20)

    assert all(w.train[0] == 0 for w in windows)
    assert [w.train_size for w in windows] == [40, 60, 80]


def test_windows_never_look_ahead():
    for window in anchored_windows(200, 50, 25):
        assert window.train.max() < window.test.min()


def test_test_windows_are_disjoint_by_default():
    windows = rolling_windows(200, 50, 25)
    seen = set()

    for window in windows:
        assert not seen.intersection(window.test.tolist())
        seen.update(window.test.tolist())


def test_purge_drops_training_bars_reaching_into_the_test_set():
    # Bars 95..99 would have labels extending past the end of the test block.
    train = np.arange(90, 105)
    test = np.arange(100, 110)

    kept = purge_train_indices(train, test, embargo=5)

    assert 99 not in kept.tolist()
    assert 100 not in kept.tolist()
    assert kept.max() == 94


def test_purge_is_a_no_op_without_an_embargo():
    train = np.arange(90, 100)
    test = np.arange(100, 110)

    assert np.array_equal(purge_train_indices(train, test, embargo=0), train)


def test_purged_walk_forward_removes_leakage_at_every_window():
    embargo = 5
    windows = purged_walk_forward(200, 60, 20, embargo=embargo)

    for window in windows:
        test_lo, test_hi = window.test.min(), window.test.max()
        for bar in window.train:
            # Either the whole label window clears the test block, or it does not overlap.
            assert (bar + embargo < test_lo) or (bar > test_hi)


def test_purged_walk_forward_shrinks_training_sets_relative_to_raw():
    raw = anchored_windows(200, 60, 20)
    purged = purged_walk_forward(200, 60, 20, embargo=5)

    assert purged[-1].train_size < raw[-1].train_size


def test_purged_kfold_covers_every_bar_exactly_once():
    windows = purged_kfold(120, 5)

    seen = np.concatenate([w.test for w in windows])
    assert sorted(seen.tolist()) == list(range(120))


def test_combinatorial_paths_enumerate_every_combination():
    assert len(combinatorial_paths(6, 2)) == 15
    assert len(combinatorial_paths(6, 3)) == 20

    with pytest.raises(ValueError):
        combinatorial_paths(4, 4)


def test_combinatorial_split_builds_one_path_per_combination():
    windows = combinatorial_purged_split(120, 6, 2, embargo=2)

    assert len(windows) == 15
    for window in windows:
        assert np.intersect1d(window.train, window.test).size == 0
        # Training and testing partition the series exactly once; purging then
        # removes the training bars whose labels reached into the test blocks,
        # so the union is a subset of the series rather than all of it.
        combined = np.sort(np.concatenate([window.train, window.test]))
        assert combined.size == np.unique(combined).size
        assert np.all(combined < 120)


def test_combinatorial_split_refuses_too_few_observations():
    with pytest.raises(ValueError):
        combinatorial_purged_split(3, 6, 2)


def test_stitching_counts_each_bar_once_and_orders_chronologically():
    windows = rolling_windows(100, 40, 20)
    returns = np.arange(100, dtype=np.float64)

    stitched = stitch_oos_returns(windows, returns)

    expected_bars = sorted(set().union(*[set(w.test.tolist()) for w in windows]))
    assert stitched.size == len(expected_bars)
    assert list(stitched) == sorted(expected_bars)


def test_stitching_rejects_out_of_range_indices():
    with pytest.raises(IndexError):
        stitch_oos_returns(rolling_windows(10, 4, 4), np.zeros(5))


def test_performance_metrics_on_a_constant_loss():
    metrics = performance_metrics(np.full(10, -0.01), periods_per_year=365)

    assert metrics['observations'] == 10
    assert metrics['total_return'] < 0
    assert metrics['hit_rate'] == 0.0
    assert metrics['max_drawdown'] < 0


def test_performance_metrics_survive_an_empty_series():
    metrics = performance_metrics(np.array([]))

    assert metrics['observations'] == 0
    assert metrics['total_return'] == 0.0


def test_performance_metrics_do_not_divide_by_zero_on_constant_returns():
    metrics = performance_metrics(np.zeros(10))

    assert np.isnan(metrics['sharpe'])
    assert metrics['total_return'] == 0.0


def test_evaluate_windows_scores_the_stitched_series_not_the_best_path():
    windows = purged_walk_forward(200, 60, 20, embargo=2)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 200)

    report = evaluate_windows(windows, returns)

    assert report['path_count'] == len(windows)
    assert report['stitched']['observations'] == len(report['oos_returns'])
    assert 0.0 <= report['profitable_path_ratio'] <= 1.0
    spread = report['spread']['sharpe']
    assert spread['min'] <= spread['median'] <= spread['max']


def test_evaluate_windows_requires_windows():
    with pytest.raises(ValueError):
        evaluate_windows([], np.zeros(10))


def test_invalid_window_arguments_are_rejected():
    with pytest.raises(ValueError):
        rolling_windows(100, 0, 10)
    with pytest.raises(ValueError):
        rolling_windows(100, 10, 10, step=0)
    with pytest.raises(ValueError):
        purged_kfold(10, 1)