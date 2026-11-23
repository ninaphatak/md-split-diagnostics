"""Diagnostics for time-correlated per-frame features.

Shape conventions used throughout:
    x         : (n_frames, n_features) array; a 1-D array is one feature.
    traj_ids  : (n_frames,) non-decreasing int array, or None for a single
                trajectory. No lag, half, or neighbourhood may span two
                trajectories.
    returns   : per-feature arrays of shape (n_features,).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, kstwobign

__all__ = [
    "SplitReport",
    "adjacent_fraction",
    "drift_test",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "split_test",
    "validate_split",
]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
    return x


def _as_traj_ids(traj_ids: np.ndarray | None, n_frames: int) -> np.ndarray:
    if traj_ids is None:
        return np.zeros(n_frames, dtype=int)
    traj_ids = np.asarray(traj_ids)
    if traj_ids.shape != (n_frames,):
        raise ValueError(
            f"traj_ids must have shape ({n_frames},), got {traj_ids.shape}"
        )
    return traj_ids


def _pooled_autocovariance(x: np.ndarray, traj_ids: np.ndarray) -> np.ndarray:
    """Autocovariance per lag and feature, summed within trajectories.

    Returns an array of shape ``(longest_trajectory, n_features)``. Products
    are only formed between frames of the same trajectory; the sum at every
    lag is divided by the total number of frames (the usual biased
    estimator, which keeps the noisy long lags small).
    """
    n_frames, n_features = x.shape
    centered = x - x.mean(axis=0)
    segments = [centered[traj_ids == t] for t in np.unique(traj_ids)]
    acov = np.zeros((max(len(s) for s in segments), n_features))
    for seg in segments:
        m = len(seg)
        nfft = 1 << (2 * m - 1).bit_length()  # zero-pad so lags do not wrap
        spectrum = np.fft.rfft(seg, n=nfft, axis=0)
        acov[:m] += np.fft.irfft(spectrum * spectrum.conj(), n=nfft, axis=0)[:m]
    return acov / n_frames


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
    and the estimates are combined before windowing. Every trajectory is
    centred on the mean of the whole data set, so trajectories that sit at
    different levels show up as a long correlation time rather than being
    hidden.

    The estimate is only trustworthy when each trajectory is much longer
    than tau_int (a common rule of thumb is 50 times longer).

    Parameters
    ----------
    x : (n_frames,) or (n_frames, n_features) array
    traj_ids : (n_frames,) int array or None
    c : float
        Window constant; 5 is Sokal's usual choice.

    Returns
    -------
    tau : (n_features,) float array
        NaN for a feature that never varies.
    """
    x = _as_2d(x)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 frames to estimate an autocorrelation time")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    traj_ids = _as_traj_ids(traj_ids, x.shape[0])

    acov = _pooled_autocovariance(x, traj_ids)
    lags = np.arange(acov.shape[0])
    tau = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        if acov[0, j] <= 0:
            continue
        # running[M] = 1/2 + sum_{k=1..M} rho(k)
        running = np.cumsum(acov[:, j] / acov[0, j]) - 0.5
        ok = lags >= c * running
        window = int(np.argmax(ok)) if ok.any() else len(lags) - 1
        tau[j] = running[window]
    return tau


def effective_sample_size(
    x: np.ndarray,
    traj_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Effective number of independent samples per feature: n / (2 * tau_int).

    tau_int is floored at 0.5 (its value for uncorrelated data), so the
    result never exceeds the number of frames.

    Returns
    -------
    ess : (n_features,) float array
        NaN for a feature that never varies.
    """
    x = _as_2d(x)
    tau = integrated_autocorrelation_time(x, traj_ids)
    return x.shape[0] / (2.0 * np.maximum(tau, 0.5))


def drift_test(
    x: np.ndarray,
    traj_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample KS test between the first and second half of each feature.

    With several trajectories, each trajectory is split at its own midpoint,
    and the pooled first halves are compared with the pooled second halves.

    The p-values treat frames as independent, so on correlated data they
    are too small; compare the statistics between features rather than
    reading the p-values literally.

    Returns
    -------
    statistic : (n_features,) float array
    pvalue : (n_features,) float array
    """
    x = _as_2d(x)
    traj_ids = _as_traj_ids(traj_ids, x.shape[0])

    first_idx = []
    second_idx = []
    for trajectory in np.unique(traj_ids):
        rows = np.flatnonzero(traj_ids == trajectory)
        half = len(rows) // 2
        first_idx.append(rows[:half])
        second_idx.append(rows[half:])
    first_idx = np.concatenate(first_idx)
    second_idx = np.concatenate(second_idx)

    statistic = np.empty(x.shape[1], dtype=float)
    pvalue = np.empty(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        statistic[j], pvalue[j] = ks_2samp(x[first_idx, j], x[second_idx, j])

    return statistic, pvalue


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
    traj_ids = _as_traj_ids(traj_ids, n_frames)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

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
            has_train_neighbor[valid] |= is_train_neighbor & same

    return float(np.mean(has_train_neighbor))


def split_test(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    traj_ids: np.ndarray | None = None,
    window: int = 1,
) -> dict:
    """Compare the train and test distributions, and measure temporal adjacency.

    The KS p-values treat frames as independent, so on correlated data they
    are too small. ``validate_split`` corrects them for the correlation time.

    Returns
    -------
    dict with keys
        "ks_statistic" : (n_features,) array
        "ks_pvalue"    : (n_features,) array
        "adjacent_fraction" : float, fraction of test frames with at least
            one train frame within ``window`` frames in the same trajectory.
    """
    x = _as_2d(x)
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    statistic = np.empty(x.shape[1])
    pvalue = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        statistic[j], pvalue[j] = ks_2samp(x[train_idx, j], x[test_idx, j])

    return {
        "ks_statistic": statistic,
        "ks_pvalue": pvalue,
        "adjacent_fraction": adjacent_fraction(
            len(x), train_idx, test_idx, traj_ids, window
        ),
    }


@dataclass(frozen=True)
class SplitReport:
    """Outcome of ``validate_split``.

    Attributes
    ----------
    passed : bool
        True if the split passed every check.
    limiting_feature : str or None
        The feature behind the first failed check; None if the split passed.
    reason : str
        One sentence per failed check, or a short note that the split passed.
    feature_names : list of str
    tau_int : (n_features,) array
        Integrated autocorrelation time per feature, in frames.
    window : int
        Frames either side of a test frame that must hold no train frame.
    adjacent_fraction : float
        Fraction of test frames with a train frame inside ``window``.
    ks_statistic : (n_features,) array
        KS distance between the train and test distributions.
    ks_pvalue : (n_features,) array
        KS p-value computed from effective, not raw, sample sizes.
    alpha : float
        Significance level, shared across features (Bonferroni).
    """

    passed: bool
    limiting_feature: str | None
    reason: str
    feature_names: list[str]
    tau_int: np.ndarray
    window: int
    adjacent_fraction: float
    ks_statistic: np.ndarray
    ks_pvalue: np.ndarray
    alpha: float

    def __str__(self) -> str:
        width = max(len(name) for name in self.feature_names + ["feature"])
        lines = [
            f"Split {'PASSED' if self.passed else 'FAILED'}: {self.reason}",
            f"  leakage: {self.adjacent_fraction:.1%} of test frames have a train "
            f"frame within {self.window} frame(s)",
            f"  {'feature':<{width}}  {'tau_int':>10}  {'KS stat':>8}  {'KS p':>9}",
        ]
        for name, tau, stat, p in zip(
            self.feature_names, self.tau_int, self.ks_statistic, self.ks_pvalue, strict=True
        ):
            lines.append(f"  {name:<{width}}  {tau:>10.1f}  {stat:>8.3f}  {p:>9.2g}")
        return "\n".join(lines)


def validate_split(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    traj_ids: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    gap_factor: float = 3.0,
    alpha: float = 0.05,
) -> SplitReport:
    """Combine the diagnostics into a pass/fail ``SplitReport``.

    The report names the limiting feature and gives the reason it failed.
    Two checks are made:

    Leakage
        No test frame may have a train frame within ``window`` frames in the
        same trajectory, where ``window = round(gap_factor * (tau - 1/2))``
        and tau is the largest tau_int over the features. ``tau - 1/2`` is
        the summed autocorrelation, which is 0 for uncorrelated data, so
        independent frames need no gap at all. The limiting feature is the
        slowest one.

    Distribution shift
        The train and test distributions of every feature are compared with
        a two-sample KS test. Frames are correlated, so the p-value uses
        effective sample sizes ``n / (2 * tau_int)`` in place of the raw
        frame counts. The split fails if any p-value is below
        ``alpha / n_features``. The limiting feature is the one with the
        smallest p-value.

    tau_int is estimated from all of ``x``, not from the train or test
    frames alone.

    Parameters
    ----------
    x : (n_frames,) or (n_frames, n_features) array
    train_idx, test_idx : int arrays
        Frame indices of the train and test sets; they must not overlap.
    traj_ids : (n_frames,) int array or None
    feature_names : list of str or None
        Defaults to ``f0, f1, ...``.
    gap_factor : float
        Required train/test separation in units of the correlation time.
    alpha : float
        Significance level for the distribution-shift check.

    Returns
    -------
    SplitReport
    """
    x = _as_2d(x)
    n_frames, n_features = x.shape
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train_idx and test_idx must both be non-empty")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train_idx and test_idx overlap")
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(n_features)]
    feature_names = list(feature_names)
    if len(feature_names) != n_features:
        raise ValueError(
            f"got {len(feature_names)} feature names for {n_features} features"
        )

    tau = integrated_autocorrelation_time(x, traj_ids)
    varying = np.isfinite(tau)
    failures = []
    limiting = None

    # Leakage: are test frames separated from train frames by the correlation time?
    window = 0
    slowest = None
    if varying.any():
        slowest = int(np.nanargmax(tau))
        window = max(int(round(gap_factor * (tau[slowest] - 0.5))), 0)
    adjacent = adjacent_fraction(n_frames, train_idx, test_idx, traj_ids, window)
    if adjacent > 0:
        limiting = feature_names[slowest]
        failures.append(
            f"{adjacent:.1%} of test frames have a train frame within {window} "
            f"frame(s), which is {gap_factor:g} correlation times of "
            f"'{limiting}' (tau_int = {tau[slowest]:.1f} frames)"
        )

    # Distribution shift: KS test with effective sample sizes.
    statistic = np.zeros(n_features)
    pvalue = np.ones(n_features)
    for j in np.flatnonzero(varying):
        statistic[j] = ks_2samp(x[train_idx, j], x[test_idx, j]).statistic
        inefficiency = 2.0 * max(tau[j], 0.5)
        n_train = train_idx.size / inefficiency
        n_test = test_idx.size / inefficiency
        scale = np.sqrt(n_train * n_test / (n_train + n_test))
        pvalue[j] = kstwobign.sf(statistic[j] * scale)
    worst = int(np.argmin(pvalue))
    threshold = alpha / n_features
    if pvalue[worst] < threshold:
        if limiting is None:
            limiting = feature_names[worst]
        failures.append(
            f"train and test distributions of '{feature_names[worst]}' differ "
            f"(KS statistic = {statistic[worst]:.3f}, p = {pvalue[worst]:.2g} "
            f"< {threshold:.2g})"
        )

    return SplitReport(
        passed=not failures,
        limiting_feature=limiting,
        reason="; ".join(failures) if failures else "no leakage or distribution shift found",
        feature_names=feature_names,
        tau_int=tau,
        window=window,
        adjacent_fraction=adjacent,
        ks_statistic=statistic,
        ks_pvalue=pvalue,
        alpha=alpha,
    )
