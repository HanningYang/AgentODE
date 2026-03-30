"""Scaling-related statistics for time series."""

from typing import Optional

import numpy as np
from scipy import stats


def dfa_alpha(
    x: np.ndarray,
    min_scale: int = 4,
    max_scale: Optional[int] = None,
    n_scales: int = 10,
) -> float:
    """Estimate the detrended fluctuation analysis (DFA) scaling exponent."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < min_scale * 4:
        return np.nan
    if max_scale is None:
        max_scale = n // 4
    if max_scale < min_scale:
        return np.nan

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
    y = np.cumsum(x - np.mean(x))
    flucts = []
    valid_scales = []

    for s in scales:
        n_windows = n // s
        if n_windows < 2:
            continue
        f_sq = []
        t = np.arange(s)
        for w in range(n_windows):
            seg = y[w * s : (w + 1) * s]
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            f_sq.append(float(np.mean((seg - trend) ** 2)))
        flucts.append(float(np.sqrt(np.mean(f_sq))))
        valid_scales.append(s)

    if len(valid_scales) < 3:
        return np.nan

    lr = stats.linregress(np.log(valid_scales), np.log(flucts))
    return float(lr.slope)


def spectral_slope(x: np.ndarray) -> float:
    """Estimate the slope of log power versus log frequency in the power spectrum."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n)
    mask = freqs > 0
    if mask.sum() <= 3:
        return np.nan
    lr = stats.linregress(np.log(freqs[mask]), np.log(fft_vals[mask] + 1e-10))
    return float(lr.slope)
