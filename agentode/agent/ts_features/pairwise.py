"""Pairwise correlation statistics between two aligned time series."""

from typing import Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def _clean_pair(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two series and drop non-finite pairs."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = min(x_arr.size, y_arr.size)
    if m == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_sub = x_arr[:m]
    y_sub = y_arr[:m]
    mask = np.isfinite(x_sub) & np.isfinite(y_sub)
    return x_sub[mask], y_sub[mask]


def _is_constant(x: np.ndarray) -> bool:
    """Return True when all finite values are effectively identical."""
    if x.size == 0:
        return True
    return bool(np.allclose(x, x[0], equal_nan=False))


def diff_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between first differences of two aligned time series."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2:
        return np.nan
    dx = np.diff(x_sub)
    dy = np.diff(y_sub)
    if dx.size < 2 or _is_constant(dx) or _is_constant(dy):
        return np.nan
    corr, _ = spearmanr(dx, dy)
    return float(corr)


def diff_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between first differences."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2:
        return np.nan
    dx = np.diff(x_sub)
    dy = np.diff(y_sub)
    if dx.size < 2 or _is_constant(dx) or _is_constant(dy):
        return np.nan
    _, pval = spearmanr(dx, dy)
    return float(pval)


def abs_diff_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between absolute first differences of two series."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2:
        return np.nan
    dx = np.abs(np.diff(x_sub))
    dy = np.abs(np.diff(y_sub))
    if dx.size < 2 or _is_constant(dx) or _is_constant(dy):
        return np.nan
    corr, _ = spearmanr(dx, dy)
    return float(corr)


def abs_diff_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between absolute first differences."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2:
        return np.nan
    dx = np.abs(np.diff(x_sub))
    dy = np.abs(np.diff(y_sub))
    if dx.size < 2 or _is_constant(dx) or _is_constant(dy):
        return np.nan
    _, pval = spearmanr(dx, dy)
    return float(pval)


def level_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation between levels of two aligned series."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2 or _is_constant(x_sub) or _is_constant(y_sub):
        return np.nan
    corr, _ = spearmanr(x_sub, y_sub)
    return float(corr)


def level_corr_p_value(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute p-value for Spearman correlation between series levels."""
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 2 or _is_constant(x_sub) or _is_constant(y_sub):
        return np.nan
    _, pval = spearmanr(x_sub, y_sub)
    return float(pval)


def ccf_lag1(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman cross-correlation at lag 1: corr(x[t], y[t+1]).

    Measures how strongly x at time t predicts y one step ahead.
    """
    x_sub, y_sub = _clean_pair(x, y)
    if x_sub.size < 3:
        return np.nan
    x_lead = x_sub[:-1]
    y_lag = y_sub[1:]
    if _is_constant(x_lead) or _is_constant(y_lag):
        return np.nan
    corr, _ = spearmanr(x_lead, y_lag)
    return float(corr)
