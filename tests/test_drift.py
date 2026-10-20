import numpy as np

from md_split_diagnostics.diagnostics import drift_test


def test_flags_linear_drift_only_on_drifting_feature():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2))
    x[:, 1] += np.linspace(0, 1, 2000)
    stat, p = drift_test(x)
    assert stat.shape == p.shape == (2,)
    assert p[1] < 1e-6
    assert stat[1] > stat[0]


def test_stationary_iid_rarely_flagged():
    # With no drift, p < 0.05 should happen about 5% of the time.
    pvals = [drift_test(np.random.default_rng(s).normal(size=1000))[1][0] for s in range(100)]
    assert np.mean(np.array(pvals) < 0.05) < 0.15


def test_offset_between_trajectories_is_not_drift():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 1))
    traj_ids = np.repeat([0, 1], 1000)
    x[1000:] += 5  # traj 1 sits higher, but neither drifts within itself
    _, p = drift_test(x, traj_ids)
    assert p[0] > 0.01
    # Without traj_ids the two levels land in different halves and look like drift.
    _, p_naive = drift_test(x)
    assert p_naive[0] < 1e-10


def test_drift_within_each_trajectory_is_detected():
    rng = np.random.default_rng(2)
    traj_ids = np.repeat([0, 1, 2], 1000)
    x = rng.normal(size=(3000, 1)) + np.tile(np.linspace(0, 1, 1000), 3)[:, None]
    _, p = drift_test(x, traj_ids)
    assert p[0] < 1e-6


def test_accepts_1d_input():
    stat, p = drift_test(np.random.default_rng(0).normal(size=200))
    assert stat.shape == p.shape == (1,)
