"""Autocorrelation and spectral statistics for time series."""

from typing import Optional

import numpy as np
from statsmodels.tsa.stattools import acf


def _acf_values(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < max_lag + 1:
        max_lag = x.size - 1
    if max_lag < 1:
        return np.full(1, np.nan, dtype=float)
    return acf(x, nlags=max_lag, fft=True)


def acf_lag1(x: np.ndarray, max_lag: int = 10) -> float:
    """Compute lag-1 autocorrelation coefficient of a time series."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    return float(acf_vals[1]) if acf_vals.size > 1 else np.nan


def acf_lag2(x: np.ndarray, max_lag: int = 10) -> float:
    """Compute lag-2 autocorrelation coefficient of a time series."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    return float(acf_vals[2]) if acf_vals.size > 2 else np.nan


def acf_lag3(x: np.ndarray, max_lag: int = 10) -> float:
    """Compute lag-3 autocorrelation coefficient of a time series."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    return float(acf_vals[3]) if acf_vals.size > 3 else np.nan


def acf_lag5(x: np.ndarray, max_lag: int = 10) -> float:
    """Compute lag-5 autocorrelation coefficient of a time series."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    return float(acf_vals[5]) if acf_vals.size > 5 else np.nan


def acf_lag10(x: np.ndarray, max_lag: int = 10) -> float:
    """Compute lag-10 autocorrelation coefficient of a time series."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    return float(acf_vals[10]) if acf_vals.size > 10 else np.nan


def acf_first_zero_crossing(x: np.ndarray, max_lag: int = 20) -> float:
    """Compute the first lag where the autocorrelation crosses zero."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    if acf_vals.size < 2:
        return np.nan
    sign_changes = np.where(np.diff(np.sign(acf_vals[1:])))[0]
    return float(sign_changes[0] + 1) if sign_changes.size > 0 else np.nan


def acf_first_zero(x: np.ndarray, max_lag: int = 20) -> float:
    """Alias matching trajectory-discrepancy naming for first zero crossing."""
    return acf_first_zero_crossing(x, max_lag=max_lag)


def acf_first_minimum_lag(x: np.ndarray, max_lag: int = 20) -> float:
    """Compute the lag of the first minimum of the autocorrelation function."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    if acf_vals.size < 3:
        return np.nan
    return float(np.argmin(acf_vals[1:]) + 1)


def acf_one_over_e_timescale(x: np.ndarray, max_lag: int = 20) -> float:
    """Compute the first lag where the autocorrelation falls below 1/e."""
    acf_vals = _acf_values(x, max_lag=max_lag)
    if acf_vals.size < 2:
        return np.nan
    threshold = 1.0 / np.e
    below = np.where(acf_vals[1:] < threshold)[0]
    return float(below[0] + 1) if below.size > 0 else np.nan


def acf_1e_timescale(x: np.ndarray, max_lag: int = 20) -> float:
    """Alias for acf_one_over_e_timescale, with a shorter name for tools."""
    return acf_one_over_e_timescale(x, max_lag=max_lag)


def dominant_frequency(x: np.ndarray) -> float:
    """Compute the dominant frequency of a series via the power spectrum."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n)
    if fft_vals.size == 0:
        return np.nan
    fft_vals[0] = 0.0
    if fft_vals.sum() == 0:
        return np.nan
    return float(freqs[np.argmax(fft_vals)])


def dominant_freq(x: np.ndarray) -> float:
    """Alias matching trajectory-discrepancy naming for dominant frequency."""
    return dominant_frequency(x)


def spectral_centroid(x: np.ndarray) -> float:
    """Compute the power-weighted mean frequency of the spectrum."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n)
    if fft_vals.size == 0:
        return np.nan
    fft_vals[0] = 0.0
    total_power = fft_vals.sum()
    if total_power == 0:
        return np.nan
    return float(np.sum(freqs * fft_vals) / total_power)


def spectral_entropy(x: np.ndarray) -> float:
    """Compute Shannon entropy of the normalized power spectrum."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    if fft_vals.size == 0:
        return np.nan
    fft_vals[0] = 0.0
    total_power = fft_vals.sum()
    if total_power == 0:
        return np.nan
    p = fft_vals / total_power
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-10)))
