"""md-split-diagnostics: is your train/test split of MD frames defensible?"""

from .data import Dataset, load_alanine_dipeptide, load_features
from .diagnostics import (
    drift_test,
    effective_sample_size,
    integrated_autocorrelation_time,
    split_test,
    validate_split,
)
from .splits import blocked_split, random_split

__version__ = "0.1.0"
