import numpy as np
import pytest

from md_split_diagnostics.diagnostics import split_test
from md_split_diagnostics.splits import random_split


@pytest.fixture
def shifted():
    """Two iid features; feature 1 is shifted by +2 in the last 200 of 1000 frames."""
    x = np.random.default_rng(0).normal(size=(1000, 2))
    x[800:, 1] += 2
    return x


def test_output_keys_and_shapes(shifted):
    out = split_test(shifted, np.arange(800), np.arange(800, 1000))
    assert set(out) == {"ks_statistic", "ks_pvalue", "adjacent_fraction"}
    assert out["ks_statistic"].shape == (2,)
    assert out["ks_pvalue"].shape == (2,)


def test_detects_distribution_shift_on_right_feature(shifted):
    out = split_test(shifted, np.arange(800), np.arange(800, 1000))
    assert out["ks_pvalue"][1] < 1e-10
    assert out["ks_pvalue"][0] > 0.01


def test_random_split_mixes_shift_but_is_adjacent(shifted):
    train, test = random_split(1000, 0.2, seed=0)
    out = split_test(shifted, train, test)
    assert out["ks_pvalue"][1] > 0.01
    assert out["adjacent_fraction"] > 0.9


def test_accepts_1d_input():
    x = np.random.default_rng(0).normal(size=100)
    out = split_test(x, np.arange(80), np.arange(80, 100))
    assert out["ks_pvalue"].shape == (1,)


def test_passes_traj_ids_and_window_through():
    x = np.zeros((10, 1))
    traj_ids = np.array([0] * 5 + [1] * 5)
    out = split_test(x, np.array([5]), np.array([4]), traj_ids=traj_ids)
    assert out["adjacent_fraction"] == 0.0
    out = split_test(x, np.array([0]), np.array([3]), window=3)
    assert out["adjacent_fraction"] == 1.0
