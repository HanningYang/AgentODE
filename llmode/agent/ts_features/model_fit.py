"""Model-based statistics for time series."""

from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import acf


def ar_coefficients(x: np.ndarray, order: int = 3) -> List[float]:
    """Fit an AR model and return its coefficients [phi_1, ..., phi_order]."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < order + 5:
        return [np.nan] * order
    try:
        model = AutoReg(x, lags=order, old_names=False).fit()
        return [float(c) for c in model.params[1:]]
    except Exception:
        return [np.nan] * order


def ar_residual_variance(x: np.ndarray, order: int = 3) -> float:
    """Fit an AR model and return the unbiased variance (ddof=1) of residuals."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < order + 5:
        return np.nan
    try:
        model = AutoReg(x, lags=order, old_names=False).fit()
        resid = model.resid
        if resid.size < 2:
            return np.nan
        return float(np.var(resid, ddof=1))
    except Exception:
        return np.nan


def ar_residual_acf1(x: np.ndarray, order: int = 3) -> float:
    """Fit an AR model and return lag-1 autocorrelation of residuals."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < order + 5:
        return np.nan
    try:
        model = AutoReg(x, lags=order, old_names=False).fit()
        resid = model.resid
        if resid.size <= 5:
            return np.nan
        resid_acf = acf(resid, nlags=1, fft=True)
        return float(resid_acf[1])
    except Exception:
        return np.nan


def exp_smoothing_alpha(x: np.ndarray) -> float:
    """Fit simple exponential smoothing and return the estimated smoothing level alpha."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 5:
        return np.nan
    try:
        model = SimpleExpSmoothing(x, initialization_method="estimated").fit(
            optimized=True,
            remove_bias=True,
        )
        return float(model.params["smoothing_level"])
    except Exception:
        return np.nan


def turning_points_count(x: np.ndarray) -> float:
    """Count total turning points (local maxima and minima) in a series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)
    return float(len(peaks) + len(troughs))


def turning_point_rate(x: np.ndarray) -> float:
    """Compute the rate of turning points (local maxima and minima) per sample."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return np.nan
    count = turning_points_count(x)
    if np.isnan(count):
        return np.nan
    return float(count / n)


def ar_coef_1(x: np.ndarray, order: int = 3) -> float:
    """First autoregressive coefficient from ar_coefficients."""
    coefs = ar_coefficients(x, order=order)
    if not coefs:
        return np.nan
    return float(coefs[0])


def ar_coef_2(x: np.ndarray, order: int = 3) -> float:
    """Second autoregressive coefficient from ar_coefficients."""
    coefs = ar_coefficients(x, order=order)
    if len(coefs) < 2:
        return np.nan
    return float(coefs[1])
