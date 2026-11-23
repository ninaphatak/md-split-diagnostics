"""Validate three splits of the alanine dipeptide backbone-dihedral data.

Downloads about 6 MB into ./data on first run. Needs the examples extra:

    pip install -e ".[examples]"
    python scripts/alanine_dipeptide_demo.py
"""

import numpy as np

from md_split_diagnostics import (
    blocked_split,
    effective_sample_size,
    integrated_autocorrelation_time,
    load_alanine_dipeptide,
    random_split,
    validate_split,
)


def main() -> None:
    ds = load_alanine_dipeptide("data")
    tau = integrated_autocorrelation_time(ds.X, ds.traj_ids)
    ess = effective_sample_size(ds.X, ds.traj_ids)
    print(f"{ds.n_frames} frames in {ds.n_trajectories} trajectories")
    for name, t, e in zip(ds.feature_names, tau, ess, strict=True):
        print(f"  {name}: tau_int = {t:7.1f} frames, effective samples = {e:8.0f}")

    gap = int(np.ceil(3 * tau.max()))
    splits = {
        "random frames": random_split(ds.n_frames, 0.2, seed=0),
        "blocked, no gap": blocked_split(ds.n_frames, 0.2, gap=0, traj_ids=ds.traj_ids),
        f"blocked, gap = {gap} frames": blocked_split(
            ds.n_frames, 0.2, gap=gap, traj_ids=ds.traj_ids
        ),
    }
    for name, (train_idx, test_idx) in splits.items():
        report = validate_split(ds.X, train_idx, test_idx, ds.traj_ids, ds.feature_names)
        print(f"\n--- {name} ---\n{report}")


if __name__ == "__main__":
    main()
