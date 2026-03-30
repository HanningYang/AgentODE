"""Stationarity-related statistics for time series."""

import numpy as np


def stationarity_variation(x: np.ndarray, n_windows: int = 4) -> float:
    """Compute variation of window means normalized by global standard deviation."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2 * n_windows:
        return np.nan
    w = n // n_windows
    window_means = [float(np.mean(x[i * w : (i + 1) * w])) for i in range(n_windows)]
    global_std = float(np.std(x, ddof=1))
    if global_std == 0.0:
        return np.nan
    return float(np.std(window_means) / global_std)


def statav(x: np.ndarray, n_windows: int = 4) -> float:
    """Alias for stationarity_variation (used as 'statav' in summary stats)."""
    return stationarity_variation(x, n_windows=n_windows)


def variance_ratio(x: np.ndarray, n_windows: int = 4) -> float:
    """Compute the ratio of maximum to minimum window variance."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2 * n_windows:
        return np.nan
    w = n // n_windows
    window_vars = [float(np.var(x[i * w : (i + 1) * w], ddof=1)) for i in range(n_windows)]
    min_var = min(window_vars)
    if min_var <= 0.0:
        return np.nan
    return float(max(window_vars) / min_var)


def mean_minus_median(x: np.ndarray) -> float:
    """Compute mean minus median of a series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x) - np.median(x))


def std_first_difference(x: np.ndarray) -> float:
    """Compute the unbiased standard deviation (ddof=1) of first differences."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan
    diffs = np.diff(x)
    if diffs.size < 2:
        return np.nan
    return float(np.std(diffs, ddof=1))
