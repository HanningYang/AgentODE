"""Pairwise correlation statistics between two aligned time series."""

from typing import Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def diff_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between first differences of two aligned time series."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    dx = np.diff(x_arr[:m])
    dy = np.diff(y_arr[:m])
    if dx.size < 2:
        return np.nan
    corr, _ = spearmanr(dx, dy)
    return float(corr)


def diff_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between first differences."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    dx = np.diff(x_arr[:m])
    dy = np.diff(y_arr[:m])
    if dx.size < 2:
        return np.nan
    _, pval = spearmanr(dx, dy)
    return float(pval)


def abs_diff_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between absolute first differences of two series."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    dx = np.abs(np.diff(x_arr[:m]))
    dy = np.abs(np.diff(y_arr[:m]))
    if dx.size < 2:
        return np.nan
    corr, _ = spearmanr(dx, dy)
    return float(corr)


def abs_diff_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between absolute first differences."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    dx = np.abs(np.diff(x_arr[:m]))
    dy = np.abs(np.diff(y_arr[:m]))
    if dx.size < 2:
        return np.nan
    _, pval = spearmanr(dx, dy)
    return float(pval)


def level_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between levels of two aligned series."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    x_sub = x_arr[:m]
    y_sub = y_arr[:m]
    corr, _ = spearmanr(x_sub, y_sub)
    return float(corr)


def level_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between series levels."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m < 2:
        return np.nan
    x_sub = x_arr[:m]
    y_sub = y_arr[:m]
    _, pval = spearmanr(x_sub, y_sub)
    return float(pval)
