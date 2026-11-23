import numpy as np
import pytest
from conftest import ar1, ar1_tau_int

from md_split_diagnostics.diagnostics import (
    _pooled_autocovariance,
    effective_sample_size,
    integrated_autocorrelation_time,
)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9, 0.99])
def test_tau_matches_exact_ar1_value(phi):
    tau = integrated_autocorrelation_time(ar1(phi, 200_000, seed=1))
    assert tau.shape == (1,)
    assert tau[0] == pytest.approx(ar1_tau_int(phi), rel=0.1)


def test_tau_is_per_feature():
    x = np.column_stack([ar1(0.0, 100_000, seed=0), ar1(0.9, 100_000, seed=1)])
    tau = integrated_autocorrelation_time(x)
    assert tau.shape == (2,)
    assert tau[0] == pytest.approx(0.5, abs=0.05)
    assert tau[1] == pytest.approx(ar1_tau_int(0.9), rel=0.1)


def test_tau_pools_trajectories_of_different_length():
    lengths = [50_000, 80_000, 120_000]
    x = np.concatenate([ar1(0.9, n, seed=s) for s, n in enumerate(lengths)])
    traj_ids = np.repeat([0, 1, 2], lengths)
    tau = integrated_autocorrelation_time(x, traj_ids)
    assert tau[0] == pytest.approx(ar1_tau_int(0.9), rel=0.1)


def test_trajectories_at_different_levels_look_slow():
    # Each trajectory is white noise, but the second sits 5 units higher. The
    # data have not converged, so tau must be long rather than 0.5.
    x = np.random.default_rng(0).normal(size=20_000)
    x[10_000:] += 5
    traj_ids = np.repeat([0, 1], 10_000)
    assert integrated_autocorrelation_time(x, traj_ids)[0] > 100


def test_autocovariance_does_not_cross_trajectory_boundary():
    x = np.array([[1.0], [3.0], [-2.0], [-2.0]])  # mean 0
    one_traj = _pooled_autocovariance(x, np.zeros(4, dtype=int))
    two_traj = _pooled_autocovariance(x, np.array([0, 0, 1, 1]))
    # Lag 1: 1*3 + 3*(-2) + (-2)*(-2) in one trajectory; the 3*(-2) pair
    # straddles the boundary and must be dropped with two.
    assert one_traj[1, 0] == pytest.approx((3 - 6 + 4) / 4)
    assert two_traj[1, 0] == pytest.approx((3 + 4) / 4)
    assert two_traj.shape == (2, 1)


def test_tau_of_constant_feature_is_nan():
    x = np.column_stack([np.ones(1000), ar1(0.5, 1000)])
    tau = integrated_autocorrelation_time(x)
    assert np.isnan(tau[0])
    assert np.isfinite(tau[1])


def test_tau_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        integrated_autocorrelation_time(np.array([1.0]))
    with pytest.raises(ValueError, match="c must be positive"):
        integrated_autocorrelation_time(np.arange(10.0), c=0)
    with pytest.raises(ValueError, match="traj_ids"):
        integrated_autocorrelation_time(np.arange(10.0), traj_ids=np.zeros(9, dtype=int))


def test_ess_is_n_over_two_tau():
    x = ar1(0.9, 200_000, n_features=2, seed=3)
    tau = integrated_autocorrelation_time(x)
    np.testing.assert_allclose(effective_sample_size(x), 200_000 / (2 * tau))


def test_ess_of_iid_data_is_about_n_and_never_more():
    ess = effective_sample_size(np.random.default_rng(0).normal(size=(50_000, 3)))
    assert np.all(ess <= 50_000)
    assert np.all(ess > 0.9 * 50_000)
