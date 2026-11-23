import numpy as np
import pytest
from conftest import ar1

from md_split_diagnostics import SplitReport, blocked_split, random_split, validate_split

N = 50_000


@pytest.fixture(scope="module")
def correlated():
    """Two stationary AR(1) features; 'slow' has tau_int = 9.5 frames."""
    return np.column_stack([ar1(0.5, N, seed=0), ar1(0.9, N, seed=1)])


def test_random_split_of_correlated_data_fails_on_leakage(correlated):
    train, test = random_split(N, 0.2, seed=0)
    report = validate_split(correlated, train, test, feature_names=["fast", "slow"])
    assert isinstance(report, SplitReport)
    assert not report.passed
    assert report.limiting_feature == "slow"
    assert "slow" in report.reason and "train frame within" in report.reason
    assert report.adjacent_fraction > 0.9
    assert report.window == round(3 * (report.tau_int[1] - 0.5))


def test_blocked_split_with_enough_gap_passes(correlated):
    train, test = blocked_split(N, 0.2, gap=100)
    report = validate_split(correlated, train, test)
    assert report.passed
    assert report.limiting_feature is None
    assert report.adjacent_fraction == 0.0
    assert report.feature_names == ["f0", "f1"]


def test_blocked_split_with_too_small_gap_fails(correlated):
    train, test = blocked_split(N, 0.2, gap=5)
    report = validate_split(correlated, train, test)
    assert not report.passed
    assert report.limiting_feature == "f1"


def test_gap_factor_sets_the_required_separation(correlated):
    train, test = blocked_split(N, 0.2, gap=5)
    assert validate_split(correlated, train, test, gap_factor=0.5).passed


def test_random_split_of_independent_frames_passes():
    x = np.random.default_rng(0).normal(size=(N, 2))
    train, test = random_split(N, 0.2, seed=0)
    report = validate_split(x, train, test)
    assert report.passed
    assert report.window == 0


def test_distribution_shift_names_the_shifted_feature(correlated):
    x = correlated.copy()
    x[int(0.8 * N):, 0] += 1.0  # only the test block of 'fast' moves
    train, test = blocked_split(N, 0.2, gap=100)
    report = validate_split(x, train, test, feature_names=["fast", "slow"])
    assert not report.passed
    assert report.limiting_feature == "fast"
    assert "distributions of 'fast' differ" in report.reason
    assert np.argmin(report.ks_pvalue) == 0


def test_correlation_alone_is_not_reported_as_shift(correlated):
    # A naive KS test rejects far too often on correlated frames; the
    # effective-sample-size correction must not.
    rejections = 0
    for seed in range(40):
        x = ar1(0.95, 20_000, seed=seed)
        train, test = blocked_split(20_000, 0.2, gap=200)
        rejections += not validate_split(x, train, test).passed
    assert rejections <= 6


def test_constant_feature_is_ignored(correlated):
    x = np.column_stack([correlated, np.ones(N)])
    train, test = blocked_split(N, 0.2, gap=100)
    report = validate_split(x, train, test)
    assert report.passed
    assert np.isnan(report.tau_int[2])
    assert report.ks_pvalue[2] == 1.0


def test_trajectory_boundaries_are_respected():
    x = np.concatenate([ar1(0.9, 10_000, seed=s) for s in range(3)])
    traj_ids = np.repeat([0, 1, 2], 10_000)
    train, test = blocked_split(30_000, 0.2, gap=100, traj_ids=traj_ids)
    assert validate_split(x, train, test, traj_ids=traj_ids).passed
    # Read as one long trajectory, each test block touches the next train block.
    assert not validate_split(x, train, test).passed


def test_report_prints_verdict_and_table(correlated):
    train, test = random_split(N, 0.2, seed=0)
    text = str(validate_split(correlated, train, test, feature_names=["fast", "slow"]))
    assert text.startswith("Split FAILED")
    assert "tau_int" in text and "slow" in text
    train, test = blocked_split(N, 0.2, gap=100)
    assert str(validate_split(correlated, train, test)).startswith("Split PASSED")


def test_rejects_bad_input(correlated):
    train, test = blocked_split(N, 0.2, gap=100)
    with pytest.raises(ValueError, match="overlap"):
        validate_split(correlated, train, train[:10])
    with pytest.raises(ValueError, match="non-empty"):
        validate_split(correlated, train, np.array([], dtype=int))
    with pytest.raises(ValueError, match="feature names"):
        validate_split(correlated, train, test, feature_names=["only_one"])
