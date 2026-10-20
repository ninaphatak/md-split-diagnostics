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
from scipy.stats import ks_2samp


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
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_features = x.shape[1]
    n_frames = len(x)
    statistic = np.empty(n_features)
    pvalue = np.empty(n_features)

    for j in range (n_features):
        statistic[j], pvalue[j] = ks_2samp(x[train_idx][:, j], x[test_idx][:, j])

    adjacent = adjacent_fraction(n_frames, train_idx, test_idx, traj_ids, window)

    dict_out = {
        "ks_statistic": statistic,
        "ks_pvalue": pvalue,
        "adjacent_fraction": adjacent, 
    }
    return dict_out


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


def adjacent_fraction(
    n_frames: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    traj_ids: np.ndarray | None = None,
    window: int = 1,
) -> float:
    """Fraction of test frames that have a train frame within ``window`` frames.

    Neighbours are only counted inside the same trajectory, so frames on
    either side of a trajectory boundary are not adjacent. A random
    frame-level split gives a value near 1 (leaky); a blocked split with
    ``gap >= window`` gives 0.

    Parameters
    ----------
    n_frames : int
        Total number of frames (rows of x).
    train_idx, test_idx : int arrays
        Frame indices of the train and test sets.
    traj_ids : (n_frames,) int array or None
        Trajectory label per frame; None means a single trajectory.
    window : int
        How many frames before and after each test frame to check.

    Returns
    -------
    float in [0, 1]; 0.0 if there are no test frames.
    """
    if traj_ids is None:
        traj_ids = np.zeros(n_frames, dtype=int)
    
    if test_idx.size == 0:
        return 0.0
    
    is_train = np.zeros(n_frames, dtype=bool)
    is_train[train_idx] = True

    has_train_neighbor = np.zeros(len(test_idx), dtype=bool)
    for s in range(1, window + 1):
        for shift in (s, -s):
            nb = test_idx + shift
            valid = (nb >= 0) & (nb < n_frames)
            is_train_neighbor = is_train[nb[valid]]
            same = traj_ids[nb[valid]] == traj_ids[test_idx[valid]]
            hit = is_train_neighbor & same
            has_train_neighbor[valid] |= hit
    
    return float(np.mean(has_train_neighbor))