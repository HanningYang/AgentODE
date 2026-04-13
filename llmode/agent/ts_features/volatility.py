"""Volatility-related statistics for time series."""

import numpy as np


def mean_crossing_rate(x: np.ndarray) -> float:
    """Fraction of time the series crosses its mean level."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan
    mean_val = float(np.mean(x))
    return float(np.mean(np.diff(np.sign(x - mean_val)) != 0))


def mean_abs_diff(x: np.ndarray) -> float:
    """Mean absolute first difference of the series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan
    diffs = np.diff(x)
    return float(np.mean(np.abs(diffs)))

