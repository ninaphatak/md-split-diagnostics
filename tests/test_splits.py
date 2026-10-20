import numpy as np
import pytest

from md_split_diagnostics.splits import blocked_split, random_split


def test_random_split_sizes_and_partition():
    train, test = random_split(1000, test_frac=0.2, seed=0)
    assert len(test) == 200
    assert len(train) == 800
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(np.union1d(train, test), np.arange(1000))


def test_random_split_returns_sorted():
    train, test = random_split(1000, test_frac=0.2, seed=0)
    assert np.all(np.diff(train) > 0)
    assert np.all(np.diff(test) > 0)


def test_random_split_is_actually_random():
    _, test = random_split(1000, test_frac=0.2, seed=0)
    assert not np.array_equal(test, np.arange(200))


def test_random_split_seed_reproducible():
    a = random_split(1000, test_frac=0.2, seed=7)
    b = random_split(1000, test_frac=0.2, seed=7)
    c = random_split(1000, test_frac=0.2, seed=8)
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_random_split_rounds_test_size():
    _, test = random_split(100, test_frac=0.29, seed=0)
    assert len(test) == 29


def test_blocked_split_layout():
    train, test = blocked_split(10, test_frac=0.2, gap=2)
    np.testing.assert_array_equal(train, np.arange(0, 6))
    np.testing.assert_array_equal(test, np.array([8, 9]))


def test_blocked_split_no_gap():
    train, test = blocked_split(10, test_frac=0.2, gap=0)
    np.testing.assert_array_equal(train, np.arange(0, 8))
    np.testing.assert_array_equal(test, np.array([8, 9]))


@pytest.mark.parametrize("gap", [0, 3, 17, 42.5])
def test_blocked_split_respects_gap(gap):
    train, test = blocked_split(1000, test_frac=0.2, gap=gap)
    assert np.intersect1d(train, test).size == 0
    assert test[0] - train[-1] - 1 >= gap
    assert len(test) == 200


def test_blocked_split_float_gap_gives_int_indices():
    train, test = blocked_split(100, test_frac=0.2, gap=4.3)
    assert train.dtype.kind == "i"
    assert test.dtype.kind == "i"
    assert test[0] - train[-1] - 1 == 5  # rounded up


@pytest.mark.parametrize("gap", [-1, -0.5])
def test_blocked_split_rejects_negative_gap(gap):
    with pytest.raises(ValueError, match="gap"):
        blocked_split(10, test_frac=0.2, gap=gap)


@pytest.mark.parametrize("test_frac, gap", [(0.2, 8), (0.2, 20), (0.01, 0)])
def test_blocked_split_rejects_empty_sets(test_frac, gap):
    with pytest.raises(ValueError, match="Not enough frames"):
        blocked_split(10, test_frac=test_frac, gap=gap)


TRAJ_IDS = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])


def test_blocked_split_per_trajectory():
    train, test = blocked_split(12, test_frac=0.25, gap=1, traj_ids=TRAJ_IDS)
    np.testing.assert_array_equal(train, [0, 1, 4, 7, 8, 9])
    np.testing.assert_array_equal(test, [3, 6, 11])


def test_blocked_split_every_trajectory_gets_test_frames():
    traj_ids = np.repeat([0, 1, 2], 1000)
    train, test = blocked_split(3000, test_frac=0.2, gap=10, traj_ids=traj_ids)
    for t in range(3):
        assert np.sum(traj_ids[test] == t) == 200
    assert np.intersect1d(train, test).size == 0


def test_blocked_split_arbitrary_traj_labels():
    a = blocked_split(12, test_frac=0.25, traj_ids=TRAJ_IDS)
    b = blocked_split(12, test_frac=0.25, traj_ids=TRAJ_IDS * 10 + 5)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_blocked_split_rejects_traj_ids_length_mismatch():
    with pytest.raises(ValueError, match="traj_ids"):
        blocked_split(10, test_frac=0.2, traj_ids=TRAJ_IDS)
