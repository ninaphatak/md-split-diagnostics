---
name: md-split-diagnostics
description: Check whether a train/test split of time-correlated molecular dynamics frames is defensible before training or scoring a model on it. Use when splitting per-frame MD features (dihedrals, distances, learned embeddings) from one or more trajectories for ML, when choosing between a random and a time-blocked split, or when a test score on simulation data looks too good.
---

# md-split-diagnostics

Consecutive MD frames are strongly correlated, so a random frame-level split scores a model on near-copies of its training frames. Use the `md_split_diagnostics` package to build splits that respect time and trajectory boundaries and to measure how leaky a split is.

## Setup

```bash
pip install git+https://github.com/ninaphatak/md-split-diagnostics
```

## Inputs

- A feature matrix `X` of shape `(n_frames, n_features)`, feature names, and an integer `traj_ids` array with one value per frame. Frames of one trajectory must be contiguous and in time order (`traj_ids` non-decreasing). Wrap them in `Dataset(X, feature_names, traj_ids)`.
- For a single trajectory saved as `.npy` or `.csv`, use `load_features(path)`.
- Pass dihedral angles (radians) through `dihedrals_to_features(angles, names)` first, to remove the ±π wraparound.
- `load_alanine_dipeptide()` gives a demo dataset (mdshare, 3 × 250 ns).

## Steps

1. Load the features into a `Dataset`, with one `traj_ids` value per trajectory. Never let frames from different trajectories be treated as neighbours.
2. Run `integrated_autocorrelation_time(ds.X, ds.traj_ids)` and `effective_sample_size(ds.X, ds.traj_ids)`. Report the slowest feature and how few independent samples the data holds.
3. Run `drift_test(ds.X, ds.traj_ids)`. A large KS statistic on a feature means its distribution changes within trajectories, so any time-blocked test set will differ from train on that feature.
4. Build the split the user plans to use, and a `blocked_split(ds.n_frames, test_frac, gap=3 * tau.max(), traj_ids=ds.traj_ids)` to compare it with.
5. Run `validate_split(ds.X, train_idx, test_idx, ds.traj_ids, ds.feature_names)` on each and print the report.
6. Report, for each split, whether it passed, the limiting feature and the reason. If a blocked split fails on leakage, recommend the gap that `report.window` implies.

## Outputs

- Splits: `(train_idx, test_idx)`, sorted integer index arrays.
- `integrated_autocorrelation_time`: tau_int per feature, in frames (0.5 for uncorrelated data, NaN for a constant feature).
- `effective_sample_size`: `n / (2 * tau_int)` per feature.
- `validate_split`: a `SplitReport` with `passed`, `limiting_feature`, `reason`, `tau_int`, `window`, `adjacent_fraction`, `ks_statistic` and `ks_pvalue`. `print(report)` gives a table.
- `split_test`: dict with `ks_statistic`, `ks_pvalue` (per feature) and `adjacent_fraction` (float in [0, 1]).
- `drift_test`: `(statistic, pvalue)`, per feature.

## Reading the results

- Leakage failure: the test score is inflated by near-duplicate frames. Use a blocked split or a larger gap.
- Shift failure with no leakage: the split is honest and test sits in a different region than train. Expect a lower score and say so.
- Drift flagged: those features are not equilibrated, and no split fixes that alone.

## Limitations

- tau_int is only trustworthy when each trajectory is much longer than tau_int (about 50 times). A process slower than the trajectory cannot be detected from it.
- `blocked_split` always puts the test block at the end of each trajectory.
- p-values from `split_test` and `drift_test` treat frames as independent and are too small on correlated data; treat them as relative. `validate_split` corrects for this.
- No command-line entry point; use the Python API.
