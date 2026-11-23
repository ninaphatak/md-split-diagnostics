"""md-split-diagnostics: is your train/test split of MD frames defensible?"""

from .data import Dataset, dihedrals_to_features, load_alanine_dipeptide, load_features
from .diagnostics import (
    SplitReport,
    adjacent_fraction,
    drift_test,
    effective_sample_size,
    integrated_autocorrelation_time,
    split_test,
    validate_split,
)
from .splits import blocked_split, random_split

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "SplitReport",
    "adjacent_fraction",
    "blocked_split",
    "dihedrals_to_features",
    "drift_test",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "load_alanine_dipeptide",
    "load_features",
    "random_split",
    "split_test",
    "validate_split",
]
