"""Atomic time series feature functions for use as LLM tools.

Each function computes a single statistic from one or more series, with clear
docstrings and type hints suitable for tool registration.
"""

from .distribution import mean, variance, skewness, std, iqr, outlier_fraction, outlier_frac
from .autocorrelation import (
    acf_lag1,
    acf_lag3,
    acf_1e_timescale,
    dominant_frequency,
    dominant_freq,
    acf_first_zero,
    spectral_entropy,
)
from .stationarity import statav, std_first_difference
from .entropy import permutation_entropy_m3, lempel_ziv_complexity
from .nonlinear import time_reversal_asymmetry
from .scaling import dfa_alpha, spectral_slope
from .model_fit import ar_coef_1, ar_coef_2, turning_point_rate, exp_smoothing_alpha
from .trend import spearman_trend
from .pairwise import diff_corr, level_corr
from .volatility import mean_crossing_rate, mean_abs_diff

__all__ = [
    "mean",
    "variance",
    "skewness",
    "std",
    "iqr",
    "outlier_fraction",
    "outlier_frac",
    "acf_lag1",
    "acf_lag3",
    "acf_1e_timescale",
    "dominant_frequency",
    "dominant_freq",
    "acf_first_zero",
    "spectral_entropy",
    "statav",
    "std_first_difference",
    "permutation_entropy_m3",
    "lempel_ziv_complexity",
    "time_reversal_asymmetry",
    "dfa_alpha",
    "spectral_slope",
    "ar_coef_1",
    "ar_coef_2",
    "turning_point_rate",
    "exp_smoothing_alpha",
    "spearman_trend",
    "diff_corr",
    "level_corr",
    "mean_crossing_rate",
    "mean_abs_diff",
]
