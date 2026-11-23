# md-split-diagnostics

[![CI](https://github.com/ninaphatak/md-split-diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/ninaphatak/md-split-diagnostics/actions/workflows/ci.yml)

Check whether a train/test split of molecular dynamics frames is defensible.

Consecutive MD frames are almost copies of each other. If you shuffle frames
into train and test sets, nearly every test frame has a near-duplicate in the
training set, and the test score measures memorisation rather than
generalisation. This package measures how long your features stay correlated,
builds splits that respect that time scale, and tells you whether a given split
passes.

It works on any per-frame feature matrix (dihedrals, distances, collective
variables, learned embeddings) and needs only NumPy and SciPy.

## Install

```bash
pip install git+https://github.com/ninaphatak/md-split-diagnostics
```

Python 3.10 or newer.

## Quick start

```python
import numpy as np
from md_split_diagnostics import blocked_split, integrated_autocorrelation_time, validate_split

X = np.load("features.npy")          # shape (n_frames, n_features), in time order

tau = integrated_autocorrelation_time(X)          # per feature, in frames
train_idx, test_idx = blocked_split(len(X), test_frac=0.2, gap=3 * tau.max())

report = validate_split(X, train_idx, test_idx)
print(report)
if not report.passed:
    print(report.limiting_feature, report.reason)
```

For several trajectories, concatenate them and pass `traj_ids`, an integer
label per frame. Every function then keeps lags, gaps and neighbourhoods inside
one trajectory.

## Example: alanine dipeptide

`scripts/alanine_dipeptide_demo.py` runs three splits of the 3 x 250 ns
alanine dipeptide data set from [mdshare](https://markovmodel.github.io/mdshare/)
(sin and cos of the backbone dihedrals, one frame per ps):

```text
750000 frames in 3 trajectories
  sin_phi: tau_int =   809.9 frames, effective samples =      463
  cos_phi: tau_int =     3.4 frames, effective samples =   111499
  sin_psi: tau_int =    25.3 frames, effective samples =    14803
  cos_psi: tau_int =    77.1 frames, effective samples =     4864

--- random frames ---
Split FAILED: 100.0% of test frames have a train frame within 2428 frame(s), which is 3 correlation times of 'sin_phi' (tau_int = 809.9 frames)

--- blocked, no gap ---
Split FAILED: 4.9% of test frames have a train frame within 2428 frame(s), which is 3 correlation times of 'sin_phi' (tau_int = 809.9 frames)

--- blocked, gap = 2430 frames ---
Split PASSED: no leakage or distribution shift found
```

750,000 frames hold only about 460 independent samples of `sin_phi`. The random
split fails outright, and a blocked split only passes once the gap covers the
slowest feature.

```bash
pip install -e ".[examples]"
python scripts/alanine_dipeptide_demo.py
```

## What it provides

| Function | What it does |
| --- | --- |
| `integrated_autocorrelation_time(x, traj_ids)` | Correlation time per feature, in frames, with Sokal's automatic windowing. |
| `effective_sample_size(x, traj_ids)` | Number of independent samples per feature, `n / (2 * tau_int)`. |
| `drift_test(x, traj_ids)` | KS test between the first and second half of each trajectory. |
| `random_split(n_frames, test_frac, seed)` | Frame-level shuffle; the leaky baseline. |
| `blocked_split(n_frames, test_frac, gap, traj_ids)` | Contiguous train and test blocks per trajectory, with `gap` frames dropped between them. |
| `adjacent_fraction(n_frames, train_idx, test_idx, traj_ids, window)` | Fraction of test frames with a train frame within `window` frames. |
| `split_test(x, train_idx, test_idx, traj_ids, window)` | Per-feature KS test between train and test, plus the adjacent fraction. |
| `validate_split(x, train_idx, test_idx, traj_ids, feature_names)` | Pass/fail `SplitReport` naming the limiting feature and the reason. |
| `load_features(path)` | Load a `.npy` or `.csv` feature matrix as a `Dataset`. |
| `dihedrals_to_features(angles, names)` | Turn angles into sin/cos pairs so the wraparound at pi disappears. |

## How `validate_split` decides

A split passes when both checks pass.

1. **Leakage.** No test frame may have a train frame within
   `round(gap_factor * (tau_int - 0.5))` frames in the same trajectory, using
   the slowest feature. `gap_factor` defaults to 3. `tau_int - 0.5` is zero for
   uncorrelated frames, so independent data passes with a random split.
2. **Distribution shift.** Each feature's train and test distributions are
   compared with a two-sample KS test. Correlated frames make the ordinary
   p-value far too small, so the test uses effective sample sizes
   `n / (2 * tau_int)` instead of frame counts. The split fails if any feature
   has `p < alpha / n_features` (`alpha` defaults to 0.05).

## Conventions and limits

- `tau_int = 1/2 + sum of rho(k) for k >= 1`, so uncorrelated data gives 0.5
  and the variance of a mean is `var(x) * 2 * tau_int / n`.
- Frames of one trajectory must be contiguous and in time order.
- Estimating `tau_int` needs trajectories much longer than `tau_int` itself (a
  common rule of thumb is 50 times longer). A process slower than your
  trajectory cannot be detected from that trajectory.
- With several trajectories, each is centred on the mean of all the data, so
  trajectories stuck in different states show up as a long correlation time.
- The p-values from `drift_test` and `split_test` treat frames as independent.
  Use them to rank features; use `validate_split` for a verdict.
- Periodic features such as angles should be converted with
  `dihedrals_to_features` first.

## Development

```bash
git clone https://github.com/ninaphatak/md-split-diagnostics
cd md-split-diagnostics
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT; see [LICENSE](LICENSE).
