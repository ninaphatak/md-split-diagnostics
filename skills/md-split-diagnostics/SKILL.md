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
2. Run `drift_test(ds.X, ds.traj_ids)`. A small p-value on a feature means its distribution changes within trajectories, so any time-blocked test set will differ from train on that feature.
3. Build the split the user plans to use, usually `blocked_split(ds.n_frames, test_frac, gap, traj_ids=ds.traj_ids)`, and a `random_split(ds.n_frames, test_frac, seed=0)` as the baseline.
4. Run `split_test(ds.X, train_idx, test_idx, traj_ids=ds.traj_ids, window=w)` on both, with the same `window`.
5. Report, for each split, the adjacency fraction and the per-feature KS statistics and p-values, and name the features that drift or shift.
6. Recommend `gap` and `window` larger than the slowest correlation time the user cares about, so that the blocked split's adjacency is 0.

## Outputs

- Splits: `(train_idx, test_idx)`, sorted integer index arrays.
- `split_test`: dict with `ks_statistic`, `ks_pvalue` (per feature) and `adjacent_fraction` (float in [0, 1]).
- `drift_test`: `(statistic, pvalue)`, per feature.

## Reading the results

- High adjacency: the test score is inflated by near-duplicate frames. Use a blocked split or a larger gap.
- Zero adjacency with small KS p-values: the split is honest and test sits in a different region than train. Expect a lower score and say so.
- Drift flagged: those features are not equilibrated, and no split fixes that alone.

## Limitations

- `integrated_autocorrelation_time`, `effective_sample_size` and `validate_split` are not implemented yet and raise `NotImplementedError`. Do not call them; the package cannot recommend a gap by itself, so take correlation times from the user's own analysis.
- `blocked_split` always puts the test block at the end of each trajectory.
- KS p-values on correlated frames are optimistic; treat them as relative.
- No command-line entry point; use the Python API.
