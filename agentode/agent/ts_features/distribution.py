"""Distributional statistics for one-dimensional time series."""

from typing import Optional

import numpy as np
from scipy import stats


def mean(x: np.ndarray) -> float:
    """Compute the arithmetic mean of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size > 0 else np.nan


def variance(x: np.ndarray) -> float:
    """Compute the unbiased sample variance (ddof=1) of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.var(x, ddof=1)) if x.size > 1 else np.nan


def std(x: np.ndarray) -> float:
    """Compute the unbiased sample standard deviation (ddof=1) of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else np.nan


def iqr(x: np.ndarray) -> float:
    """Interquartile range (75th - 25th percentile) of the series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def skewness(x: np.ndarray) -> float:
    """Compute Fisher-Pearson skewness of a time-ordered series."""
    import warnings
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 2:
        return np.nan
    if np.allclose(x, x[0]):
        print("[ts_features] near-constant trajectory detected — skewness returning NaN")
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.skew(x))


def kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(stats.kurtosis(x)) if x.size > 3 else np.nan


def min_value(x: np.ndarray) -> float:
    """Compute the minimum of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.min(x)) if x.size > 0 else np.nan


def max_value(x: np.ndarray) -> float:
    """Compute the maximum of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.max(x)) if x.size > 0 else np.nan


def value_range(x: np.ndarray) -> float:
    """Compute the range (max - min) of a time-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.max(x) - np.min(x)) if x.size > 0 else np.nan


def outlier_fraction(x: np.ndarray, threshold: float = 2.0) -> float:
    """Compute the fraction of values beyond a standard deviation threshold."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    mask = np.abs(x - mu) > threshold * sigma
    return float(np.mean(mask))


def outlier_frac(x: np.ndarray, threshold: float = 2.0) -> float:
    """Alias used in trajectory-discrepancy stats for outlier_fraction."""
    return outlier_fraction(x, threshold=threshold)


def distribution_entropy(x: np.ndarray, bins: Optional[int] = None) -> float:
    """Compute Shannon entropy of the empirical distribution."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if bins is None:
        bins = max(5, int(np.sqrt(x.size)))
    counts, _ = np.histogram(x, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def gaussianity_p_value(x: np.ndarray, max_n: int = 5000) -> float:
    """Compute Shapiro-Wilk test p-value for normality."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 8:
        return np.nan
    _, pval = stats.shapiro(x[:max_n])
    return float(pval)
