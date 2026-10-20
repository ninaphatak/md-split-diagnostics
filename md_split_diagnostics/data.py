"""Loading per-frame feature matrices.

Every loader returns a ``Dataset``: a feature matrix ``X`` of shape
``(n_frames, n_features)``, one name per feature, and an integer trajectory
id per frame. Frames with the same ``traj_id`` are contiguous and in time
order; frames from different trajectories are never treated as neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALANINE_FILE = "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"
ALANINE_DT_PS = 1.0  # frame spacing of the mdshare alanine dipeptide data


@dataclass
class Dataset:
    X: np.ndarray
    feature_names: list[str]
    traj_ids: np.ndarray

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.traj_ids = np.asarray(self.traj_ids, dtype=int)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length does not match X.shape[1]")
        if self.traj_ids.shape != (self.X.shape[0],):
            raise ValueError("traj_ids must have one entry per frame")
        if np.any(np.diff(self.traj_ids) < 0):
            raise ValueError("traj_ids must be non-decreasing (trajectories contiguous)")

    @property
    def n_frames(self) -> int:
        return self.X.shape[0]

    @property
    def n_trajectories(self) -> int:
        return len(np.unique(self.traj_ids))


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {arr.shape}")
    return arr


def _read_csv(path: Path) -> tuple[np.ndarray, list[str] | None]:
    with open(path) as f:
        first = f.readline().strip()
    try:
        [float(tok) for tok in first.split(",")]
        has_header = False
    except ValueError:
        has_header = True
    X = np.loadtxt(path, delimiter=",", skiprows=int(has_header), ndmin=2)
    names = [tok.strip() for tok in first.split(",")] if has_header else None
    return X, names


def load_features(path: str | Path) -> Dataset:
    """Load a single-trajectory feature matrix from ``.npy`` or ``.csv``.

    A ``.csv`` may have one header row of column names. A 1-D array is
    treated as a single feature.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        X, names = _as_2d(np.load(path, allow_pickle=False)), None
    elif suffix == ".csv":
        X, names = _read_csv(path)
    else:
        raise ValueError(f"unsupported file type {suffix!r}; use .npy or .csv")
    if names is None:
        names = [f"f{i}" for i in range(X.shape[1])]
    return Dataset(X, names, np.zeros(X.shape[0], dtype=int))


def dihedrals_to_features(angles: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Map angles (radians) to sin/cos pairs to remove the +-pi wraparound.

    ``(n, k)`` angles become ``(n, 2k)`` features ordered
    ``sin_a0, cos_a0, sin_a1, cos_a1, ...``.
    """
    angles = _as_2d(angles)
    feats = np.empty((angles.shape[0], 2 * angles.shape[1]))
    feats[:, 0::2] = np.sin(angles)
    feats[:, 1::2] = np.cos(angles)
    out_names = [f"{fn}_{n}" for n in names for fn in ("sin", "cos")]
    return feats, out_names


def load_alanine_dipeptide(data_dir: str | Path = "data") -> Dataset:
    """Download (once) and load the 3 x 250 ns alanine dipeptide phi/psi data.

    Returns sin/cos features of phi and psi for all three trajectories,
    concatenated, with ``traj_ids`` 0, 1, 2.
    """
    import mdshare

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local = mdshare.fetch(ALANINE_FILE, working_directory=str(data_dir))
    with np.load(local, allow_pickle=False) as npz:
        trajs = [npz[k] for k in sorted(npz.files)]
    blocks, ids = [], []
    for i, angles in enumerate(trajs):
        feats, names = dihedrals_to_features(angles, ["phi", "psi"])
        blocks.append(feats)
        ids.append(np.full(len(feats), i, dtype=int))
    return Dataset(np.concatenate(blocks), names, np.concatenate(ids))
