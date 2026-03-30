"""Atomic time series feature functions for use as LLM tools.

Each function computes a single statistic from one or more series, with clear
docstrings and type hints suitable for tool registration.
"""

from .distribution import mean, variance, skewness
from .autocorrelation import (
    acf_lag1,
    acf_lag3,
    acf_1e_timescale,
    dominant_frequency,
)
from .stationarity import statav, std_first_difference
from .entropy import permutation_entropy_m3
from .nonlinear import time_reversal_asymmetry
from .model_fit import ar_coef_1, ar_coef_2, turning_point_rate
from .trend import spearman_trend
from .pairwise import diff_corr

__all__ = [
    "mean",
    "variance",
    "skewness",
    "acf_lag1",
    "acf_lag3",
    "acf_1e_timescale",
    "dominant_frequency",
    "statav",
    "std_first_difference",
    "permutation_entropy_m3",
    "time_reversal_asymmetry",
    "ar_coef_1",
    "ar_coef_2",
    "turning_point_rate",
    "spearman_trend",
    "diff_corr",
]
