"""Nonlinear dynamical statistics for time series."""

import numpy as np


def time_reversal_asymmetry(x: np.ndarray) -> float:
    """Compute a time-reversal asymmetry statistic as the difference between forward and backward third-order moments."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    x_std = (x - np.mean(x)) / (np.std(x, ddof=1) + 1e-10)
    trev = float(np.mean(x_std[:-1] ** 2 * x_std[1:]) - np.mean(x_std[:-1] * x_std[1:] ** 2))
    return trev
