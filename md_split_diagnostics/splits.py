"""Train/test split constructors. Both return (train_idx, test_idx) as sorted int arrays."""

from __future__ import annotations

import numpy as np


def random_split(
    n_frames: int,
    test_frac: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each frame to test independently at random (frame-level split).

    This is the leaky baseline: on correlated data nearly every test frame
    has a train frame as an immediate neighbour.
    """
    frames = np.arange(n_frames)
    rng = np.random.default_rng(seed)
    frames = rng.permutation(frames)
    n_test = int(round(n_frames * test_frac))
    test_idx = np.sort(frames[:n_test])
    train_idx = np.sort(frames[n_test:])

    return train_idx, test_idx


def blocked_split(
    n_frames: int,
    test_frac: float = 0.2,
    gap: int = 0,
    traj_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use contiguous time blocks for train and test, with ``gap`` frames dropped between them.

    Gap frames belong to neither set. Set ``gap`` to a few tau_int. With
    several trajectories, each trajectory is split the same way, so the test
    set takes about ``test_frac`` of every trajectory.
    """

    if traj_ids is None:
        traj_ids = np.zeros(n_frames, dtype=int)

    if len(traj_ids) != n_frames:
        raise ValueError(
            f"traj_ids must have length n_frames={n_frames}, got {len(traj_ids)}"
        )

    train_indices = []
    test_indices = []
    
    for trajectory in np.unique(traj_ids):
        rows = np.flatnonzero(traj_ids == trajectory)
        train_idx, test_idx = _blocked_one(len(rows), test_frac, gap)
        train_idx += rows[0]
        test_idx += rows[0]
        train_indices.append(train_idx)
        test_indices.append(test_idx)
    
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    return train_indices, test_indices


def _blocked_one(
    n_frames: int,
    test_frac: float = 0.2,
    gap: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Use contiguous time blocks for train and test, with ``gap`` frames dropped between them.

    Gap frames belong to neither set. Set ``gap`` to a few tau_int.
    """
    n_test = int(round(n_frames * test_frac))
    train_idx = (np.arange(0, n_frames - n_test - int(np.ceil(gap))))
    test_idx = np.arange(n_frames - n_test, n_frames)

    if gap < 0:
        raise ValueError(
            f"gap must be >= 0, got {gap}"
        )

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Not enough frames for blocked split: n_frames={n_frames}, "
            f"test_frac={test_frac}, gap={gap}"
        )

    return train_idx, test_idx