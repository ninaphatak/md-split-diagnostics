import numpy as np
import pytest

from md_split_diagnostics.diagnostics import adjacent_fraction
from md_split_diagnostics.splits import blocked_split, random_split

TRAIN = np.array([0, 2, 3, 6])
TEST = np.array([1, 4, 5, 8, 9])


def test_worked_example():
    assert adjacent_fraction(10, TRAIN, TEST, window=1) == pytest.approx(0.6)


def test_does_not_wrap_around_ends():
    # Frame 0's left neighbour must not be read as frame 9 (negative index wrap).
    assert adjacent_fraction(10, np.array([9]), np.array([0]), window=1) == 0.0
    assert adjacent_fraction(10, np.array([0]), np.array([9]), window=1) == 0.0


def test_does_not_cross_trajectory_boundary():
    traj_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Test frame 4 is the last frame of traj 0; frame 5 (train) is the first of traj 1.
    assert adjacent_fraction(10, np.array([5]), np.array([4]), traj_ids=traj_ids) == 0.0
    assert adjacent_fraction(10, np.array([5]), np.array([4])) == 1.0


def test_wider_window_finds_farther_neighbours():
    train, test = np.array([0]), np.array([3])
    assert adjacent_fraction(10, train, test, window=2) == 0.0
    assert adjacent_fraction(10, train, test, window=3) == 1.0


def test_random_split_is_almost_fully_adjacent():
    train, test = random_split(10_000, test_frac=0.2, seed=0)
    assert adjacent_fraction(10_000, train, test, window=1) > 0.9


@pytest.mark.parametrize("window", [1, 5, 20])
def test_blocked_split_with_enough_gap_has_no_adjacency(window):
    traj_ids = np.repeat([0, 1, 2], 1000)
    train, test = blocked_split(3000, test_frac=0.2, gap=window, traj_ids=traj_ids)
    assert adjacent_fraction(3000, train, test, traj_ids=traj_ids, window=window) == 0.0


def test_blocked_split_with_too_small_gap_is_flagged():
    train, test = blocked_split(1000, test_frac=0.2, gap=2)
    assert adjacent_fraction(1000, train, test, window=5) > 0.0
