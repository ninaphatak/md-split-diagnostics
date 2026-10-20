import numpy as np
import pytest

from md_split_diagnostics.data import Dataset, dihedrals_to_features, load_features


def test_npy_round_trip(tmp_path, rng):
    X = rng.standard_normal((50, 3))
    path = tmp_path / "feats.npy"
    np.save(path, X)
    ds = load_features(path)
    np.testing.assert_array_equal(ds.X, X)
    assert ds.feature_names == ["f0", "f1", "f2"]
    np.testing.assert_array_equal(ds.traj_ids, np.zeros(50, dtype=int))


def test_npy_1d_is_one_feature(tmp_path, rng):
    x = rng.standard_normal(20)
    path = tmp_path / "x.npy"
    np.save(path, x)
    ds = load_features(path)
    assert ds.X.shape == (20, 1)


def test_csv_with_header(tmp_path, rng):
    X = rng.standard_normal((10, 2))
    path = tmp_path / "feats.csv"
    np.savetxt(path, X, delimiter=",", header="a,b", comments="")
    ds = load_features(path)
    np.testing.assert_allclose(ds.X, X)
    assert ds.feature_names == ["a", "b"]


def test_csv_without_header(tmp_path, rng):
    X = rng.standard_normal((10, 2))
    path = tmp_path / "feats.csv"
    np.savetxt(path, X, delimiter=",")
    ds = load_features(path)
    np.testing.assert_allclose(ds.X, X)
    assert ds.feature_names == ["f0", "f1"]


def test_unsupported_suffix(tmp_path):
    path = tmp_path / "feats.txt"
    path.write_text("1,2\n")
    with pytest.raises(ValueError, match="unsupported"):
        load_features(path)


def test_dihedrals_to_features_removes_wraparound():
    # Angles just either side of +-pi should give nearly identical features.
    angles = np.array([[np.pi - 1e-3, 0.0], [-np.pi + 1e-3, 0.0]])
    feats, names = dihedrals_to_features(angles, ["phi", "psi"])
    assert names == ["sin_phi", "cos_phi", "sin_psi", "cos_psi"]
    np.testing.assert_allclose(feats[0], feats[1], atol=3e-3)


def test_dataset_rejects_non_contiguous_traj_ids():
    with pytest.raises(ValueError, match="non-decreasing"):
        Dataset(np.zeros((3, 1)), ["f0"], np.array([0, 1, 0]))
