import numpy as np
import pytest
from scipy.signal import lfilter


def ar1(phi: float, n: int, n_features: int = 1, seed: int = 0) -> np.ndarray:
    """Stationary unit-variance AR(1): x_t = phi * x_{t-1} + sqrt(1 - phi^2) * eps_t."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, n_features)) * np.sqrt(1 - phi**2)
    eps[0] = rng.standard_normal(n_features)  # start in the stationary distribution
    return lfilter([1.0], [1.0, -phi], eps, axis=0)


def ar1_tau_int(phi: float) -> float:
    """Exact integrated autocorrelation time, convention 1/2 + sum rho(k)."""
    return (1 + phi) / (2 * (1 - phi))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
