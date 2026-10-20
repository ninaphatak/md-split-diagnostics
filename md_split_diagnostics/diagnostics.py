"""Diagnostics for time-correlated per-frame features.

Shape conventions used throughout:
    x         : (n_frames, n_features) array; a 1-D array is one feature.
    traj_ids  : (n_frames,) non-decreasing int array, or None for a single
                trajectory. No lag, half, or neighbourhood may span two
                trajectories.
    returns   : per-feature arrays of shape (n_features,).
"""

from __future__ import annotations

import numpy as np


def integrated_autocorrelation_time(
    x: np.ndarray,
    traj_ids: np.ndarray | None = None,
    c: float = 5.0,
) -> np.ndarray:
    """Integrated autocorrelation time per feature, in frames.

    Uses the convention tau_int = 1/2 + sum_{k>=1} rho(k), so an
    uncorrelated series has tau_int = 0.5 and the variance of the mean is
    var(x) * 2 * tau_int / n.

    The sum is truncated at the smallest window M with M >= c * tau_int(M)
    (Sokal's automatic windowing).

    With several trajectories, rho(k) is estimated within each trajectory
    and the estimates are combined before windowing.

    Parameters
    ----------
    x : (n_frames,) or (n_frames, n_features) array
    traj_ids : (n_frames,) int array or None
    c : float
        Window constant; 5 is Sokal's usual choice.

    Returns
    -------
    tau : (n_features,) float array
    """
    raise NotImplementedError


def effective_sample_size(
    x: np.ndarray,
    traj_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Effective number of independent samples per feature: n / (2 * tau_int).

    Returns
    -------
    ess : (n_features,) float array
    """
    raise NotImplementedError


def drift_test(
    x: np.ndarray,
    traj_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample KS test between the first and second half of each feature.

    With several trajectories, each trajectory is split at its own midpoint,
    and the pooled first halves are compared with the pooled second halves.

    Returns
    -------
    statistic : (n_features,) float array
    pvalue : (n_features,) float array
    """
    raise NotImplementedError


# --- Milestones 3-4: signatures provisional, finalised when we get there ---


def split_test(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    traj_ids: np.ndarray | None = None,
    window: int = 1,
) -> dict:
    """Compare the train and test distributions, and measure temporal adjacency.

    Returns
    -------
    dict with keys
        "ks_statistic" : (n_features,) array
        "ks_pvalue"    : (n_features,) array
        "adjacent_fraction" : float, fraction of test frames with at least
            one train frame within ``window`` frames in the same trajectory.
    """
    raise NotImplementedError


def validate_split(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    traj_ids: np.ndarray | None = None,
    feature_names: list[str] | None = None,
):
    """Combine the diagnostics into a pass/fail ``SplitReport``.

    The report names the limiting feature and gives the reason it failed.
    """
    raise NotImplementedError
