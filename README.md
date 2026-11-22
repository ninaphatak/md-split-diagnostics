# md-split-diagnostics

Check whether a train/test split of time-correlated molecular dynamics (MD) frames is defensible.

Consecutive MD frames are strongly correlated. A random frame-level split puts almost every test frame right next to a train frame from the same trajectory, so a model is scored on near-copies of its training data. This package builds splits that respect time and trajectory boundaries, and measures how leaky a given split is.

## What it checks

- **Temporal adjacency** (`adjacent_fraction`): the fraction of test frames with a train frame within `window` frames in the same trajectory. Near 1 means leaky (typical of a random split); 0 means a blocked split with `gap >= window`.
- **Train/test distribution shift** (`split_test`): a two-sample KS test per feature between train and test frames, returned with the adjacency fraction.
- **Within-trajectory drift** (`drift_test`): a two-sample KS test per feature between the first and second half of each trajectory, halves pooled across trajectories. An offset between trajectories is not counted as drift.

Split builders:

- `random_split`: frame-level random assignment, the leaky baseline.
- `blocked_split`: contiguous blocks per trajectory. The last `test_frac` of each trajectory is test, and `gap` frames before it are dropped from both sets.

## Install

```bash
pip install git+https://github.com/ninaphatak/md-split-diagnostics
```

Requires Python 3.10+.

## Usage

```python
from md_split_diagnostics import (
    Dataset, blocked_split, random_split, split_test, drift_test,
)

# X: (n_frames, n_features); traj_ids: one int per frame, contiguous per trajectory
ds = Dataset(X, feature_names, traj_ids)

stat, p = drift_test(ds.X, ds.traj_ids)

splits = {
    "random": random_split(ds.n_frames, test_frac=0.2, seed=0),
    "blocked": blocked_split(ds.n_frames, test_frac=0.2, gap=50, traj_ids=ds.traj_ids),
}
for name, (train, test) in splits.items():
    out = split_test(ds.X, train, test, traj_ids=ds.traj_ids, window=50)
    print(name, out["adjacent_fraction"], out["ks_pvalue"])
```

On correlated trajectories the random split typically shows adjacency near 1.0 with clean-looking KS p-values, while a blocked split with enough gap shows adjacency 0.0 and an honest distribution shift between train and test.

### Loading data

- `load_features(path)` reads one trajectory from `.npy` or `.csv` (an optional header row gives feature names). For several trajectories, build `Dataset(X, feature_names, traj_ids)` directly.
- `dihedrals_to_features(angles, names)` maps angles in radians to sin/cos pairs to remove the ±π wraparound.
- `load_alanine_dipeptide()` downloads the mdshare 3 × 250 ns alanine dipeptide phi/psi dataset as a demo.

### Reading the results

- High adjacency: the test score is inflated by near-duplicate frames. Use a blocked split or a larger gap.
- Zero adjacency with small KS p-values: the split is honest, and test really does sit in a different region than train. Expect a lower score.
- Drift flagged: the trajectories are not equilibrated on those features, and no split fixes that alone.

## Status

Version 0.1.0. Working: data loading, `random_split`, `blocked_split`, `adjacent_fraction`, `split_test`, `drift_test`.

Not implemented yet (exported, but they raise `NotImplementedError`):

- `integrated_autocorrelation_time`: per-feature tau_int with Sokal windowing.
- `effective_sample_size`: n / (2 tau_int).
- `validate_split`: a combined pass/fail report naming the limiting feature.

Until tau_int is available, choose `gap` and `window` from your own autocorrelation analysis.

## Limitations

- `blocked_split` always puts the test block at the end of each trajectory; there is no blocked k-fold.
- KS tests run per feature on correlated frames, so p-values are optimistic. Read them as relative, not calibrated.
- There is no command-line entry point; use the Python API.

## Claude skill

A Claude skill for this package is in [`skills/md-split-diagnostics/SKILL.md`](skills/md-split-diagnostics/SKILL.md).

## Tests

```bash
pip install -e ".[test]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
