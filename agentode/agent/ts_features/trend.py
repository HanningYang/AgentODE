"""Trend statistics for time series."""

import warnings

import numpy as np
from scipy.stats import spearmanr


def spearman_trend(x: np.ndarray, time_index: np.ndarray) -> float:
    """Compute Spearman correlation between time index and series values as a trend measure."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(time_index, dtype=float)
    mask = ~np.isnan(x)
    x = x[mask]
    t = t[mask]
    if x.size < 2:
        return np.nan
    if np.all(x == x[0]):
        print("[ts_features] constant trajectory detected — spearman_trend returning NaN")
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(t, x)
    return float(corr)


def spearman_trend_p_value(x: np.ndarray, time_index: np.ndarray) -> float:
    """Compute p-value for the Spearman trend between time and values."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(time_index, dtype=float)
    mask = ~np.isnan(x)
    x = x[mask]
    t = t[mask]
    if x.size < 2:
        return np.nan
    if np.all(x == x[0]):
        print("[ts_features] constant trajectory detected — spearman_trend_p_value returning NaN")
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pval = spearmanr(t, x)
    return float(pval)
