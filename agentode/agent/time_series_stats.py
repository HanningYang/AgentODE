"""Population-level time series summary statistics.

Implements a feature-based summary framework for longitudinal data, inspired by
Fulcher (2017). The core functions are:

  - Per-series feature extractors (distribution, autocorrelation, entropy, etc.).
  - A per-variable pipeline over a population trajectory.
  - Population-level statistics and cross-variable correlations from a DataFrame.

This module is dataset-agnostic. It assumes a long-format DataFrame with an
identifier column, a time column, and one or more numeric variables.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import ts_features as tsf


def require_length(x: np.ndarray, min_n: int, stat_name: str) -> bool:
    """Return False if series is shorter than required length."""
    return len(x) >= min_n


def make_result(value, nan_reason: Optional[str] = None) -> Tuple[float, Optional[str]]:
    """Return (value, nan_reason) tuple with no explicit NaN reason."""
    val = float(value)
    return val, None


def _nanmean(values: List[float]) -> float:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def _iqr(values: List[float]) -> float:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size < 4:
        return np.nan
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


_CATEGORY: Dict[str, str] = {
    "mean":               "distribution",
    "std":                "distribution",
    "iqr":                "distribution",
    "skewness":           "distribution",
    "outlier_frac":       "distribution",
    "value_range":        "distribution",
    "acf_lag1":           "autocorrelation",
    "acf_lag2":           "autocorrelation",
    "acf_lag3":           "autocorrelation",
    "acf_first_zero":     "autocorrelation",
    "dominant_freq":      "autocorrelation",
    "spectral_entropy":   "autocorrelation",
    "statav":             "stationarity",
    "std_first_diff":     "stationarity",
    "perm_entropy_m3":    "entropy",
    "lz_complexity":      "entropy",
    "dfa_alpha":          "scaling",
    "spectral_slope":     "scaling",
    "ar_coef_1":          "model_fit",
    "exp_smooth_alpha":   "model_fit",
    "turning_point_rate": "model_fit",
    "spearman_trend":     "trend",
    "mean_crossing_rate": "volatility",
    "mean_abs_diff":      "volatility",
    "level_corr":         "cross_variable",
    "diff_corr":          "cross_variable",
    "ccf_lag1":           "cross_variable",
}



def compute_pairwise_correlations(
    df: pd.DataFrame,
    lab_vars: List[str],
    id_col: str,
    time_col: str,
) -> Dict[str, Tuple[float, Optional[str]]]:
    """Pairwise Spearman correlations between variables based on first differences."""
    results: Dict[str, Tuple[float, Optional[str]]] = {}
    n_bio = len(lab_vars)
    grouped = df.groupby(id_col)

    for i in range(n_bio):
        for j in range(i + 1, n_bio):
            b1, b2 = lab_vars[i], lab_vars[j]
            pair_label = f"{b1}__{b2}"

            diffs1: List[float] = []
            diffs2: List[float] = []

            for _, patient_df in grouped:
                patient_df = patient_df.sort_values(time_col)
                sub = patient_df[[b1, b2]].dropna()
                v1 = sub[b1].to_numpy()
                v2 = sub[b2].to_numpy()
                m = min(v1.size, v2.size)

                if m > 1:
                    diffs1.extend(np.diff(v1[:m]).tolist())
                    diffs2.extend(np.diff(v2[:m]).tolist())

            if len(diffs1) > 1:
                arr1 = np.array(diffs1)
                arr2 = np.array(diffs2)
                corr, _ = spearmanr(arr1, arr2)
                corr_val = float(corr) if not np.isnan(corr) else 0.0
                results[f"first_diff_spearman_corr__{pair_label}"] = (corr_val, None)
            else:
                results[f"first_diff_spearman_corr__{pair_label}"] = (np.nan, None)

    return results


def compute_all_stats(
    x: np.ndarray,
    time_index: np.ndarray,
    var_name: str = "variable",
    ar_order: int = 3,
) -> Dict[str, Tuple[float, Optional[str]]]:
    """Compute all feature categories for a single time series."""
    x = np.asarray(x, dtype=float)
    time_index = np.asarray(time_index, dtype=float)

    all_stats: Dict[str, Tuple[float, Optional[str]]] = {}

    def add(category: str, name: str, value: float, reason: Optional[str] = None) -> None:
        all_stats[f"{category}__{name}"] = make_result(value, reason)

    # Distribution
    # Distribution (compact subset)
    add("distribution", "within_trajectory_mean", tsf.mean(x))
    add("distribution", "within_trajectory_variance", tsf.variance(x))
    add("distribution", "within_trajectory_skewness", tsf.skewness(x))

    # Autocorrelation (compact subset)
    # Original min: 10
    if require_length(x, 3, "ACF"):
        add("autocorrelation", "acf_lag1", tsf.acf_lag1(x))
        add("autocorrelation", "acf_lag3", tsf.acf_lag3(x))
        one_e = tsf.acf_1e_timescale(x)
        add("autocorrelation", "acf_1e_timescale", one_e)
        add("autocorrelation", "dominant_freq_normalized", tsf.dominant_frequency(x))

    # Stationarity
    # Original min: 2 * 4 = 8
    if require_length(x, 3, "stationarity"):
        add("stationarity", "window_mean_variation_normalized", tsf.statav(x))
        add("stationarity", "std_first_diff_within", tsf.std_first_difference(x))

    # Entropy and complexity
    # Original min: 10
    if require_length(x, 3, "entropy"):
        add("entropy", "perm_entropy_m3", tsf.permutation_entropy_m3(x))

    # Nonlinear
    # Original min: 10
    if require_length(x, 3, "nonlinear"):
        add("nonlinear", "time_reversal_asym", tsf.time_reversal_asymmetry(x))

    # Model fit
    # Original min: ar_order + 5
    if require_length(x, 3, "model-fit"):
        add("model_fit", "ar3_coef_1", tsf.ar_coef_1(x, order=ar_order))
        add("model_fit", "ar3_coef_2", tsf.ar_coef_2(x, order=ar_order))
        add("model_fit", "turning_point_rate", tsf.turning_point_rate(x))

    # Trend
    # Original min: 5
    if require_length(x, 3, "trend"):
        add("trend", "spearman_trend", tsf.spearman_trend(x, time_index))

    return all_stats


_PRECISION: Dict[str, int] = {
    "within_trajectory_mean": 3,
    "within_trajectory_variance": 3,
    "within_trajectory_skewness": 3,
    "acf_lag1": 4,
    "acf_lag2": 4,
    "acf_lag3": 4,
    "acf_1e_timescale": 2,
    "dominant_freq_normalized": 6,
    "window_mean_variation_normalized": 4,
    "std_first_diff_within": 3,
    "perm_entropy_m3": 4,
    "time_reversal_asym": 4,
    "ar3_coef_1": 4,
    "ar3_coef_2": 4,
    "turning_point_rate": 4,
    "spearman_trend": 4,
    "first_diff_spearman_corr": 4,
    # MNSD-style pooled per-trajectory stats
    "mean": 3,
    "std": 3,
    "iqr": 3,
    "skewness": 3,
    "outlier_frac": 4,
    "value_range": 3,
    "acf_first_zero": 2,
    "dominant_freq": 6,
    "spectral_entropy": 4,
    "statav": 4,
    "std_first_diff": 3,
    "lz_complexity": 4,
    "dfa_alpha": 4,
    "spectral_slope": 4,
    "ar_coef_1": 4,
    "exp_smooth_alpha": 4,
    "mean_crossing_rate": 4,
    "mean_abs_diff": 3,
    "level_corr": 4,
    "diff_corr": 4,
    "ccf_lag1": 4,
}
_DEFAULT_PRECISION = 4


def _round_value(statistic: str, val: float) -> float:
    """Round a value according to the precision map."""
    if np.isnan(val):
        return np.nan
    dp = _PRECISION.get(statistic, _DEFAULT_PRECISION)
    return round(val, dp)


def _row(variable: str, category: str, statistic: str, value_tuple) -> Dict[str, object]:
    """Convert a (value, nan_reason) tuple into an output row."""
    val, _ = value_tuple
    try:
        fval = float(val)
        rounded = _round_value(statistic, fval)
    except (TypeError, ValueError):
        rounded = np.nan
    return {
        "variable": variable,
        "category": category,
        "statistic": statistic,
        "value": rounded,
    }


def format_per_variable(results_dict: Dict[str, Tuple[float, Optional[str]]], var_name: str) -> pd.DataFrame:
    """Format per-variable statistics into a long DataFrame."""
    rows: List[Dict[str, object]] = []
    for full_key, value_tuple in results_dict.items():
        category, stat = full_key.split("__", 1)
        rows.append(_row(var_name, category, stat, value_tuple))
    return pd.DataFrame(rows)


def format_pairwise(results_dict: Dict[str, Tuple[float, Optional[str]]]) -> pd.DataFrame:
    """Format pairwise statistics into a long DataFrame."""
    rows: List[Dict[str, object]] = []
    for full_key, value_tuple in results_dict.items():
        parts = full_key.split("__")
        stat_type = parts[0]
        pair = "__".join(parts[1:])
        rows.append(_row(pair, "cross_variable", stat_type, value_tuple))
    return pd.DataFrame(rows)


def collect_per_trajectory_stats(
    df: pd.DataFrame,
    lab_vars: Optional[List[str]] = None,
    id_col: str = "id",
    time_col: str = "t",
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """Compute per-trajectory statistics for all variables and variable pairs.

    Returns:
        per_var: {variable: {statistic: [values...]}}
        cross:  {"var1__var2": {statistic: [values...]}}
    """
    if lab_vars is None:
        lab_vars = [c for c in df.columns if c not in [id_col, time_col]]

    per_var: Dict[str, Dict[str, List[float]]] = {var: {} for var in lab_vars}

    grouped = df.groupby(id_col)
    for _, group in grouped:
        group = group.sort_values(time_col)
        t_vals = group[time_col].to_numpy(dtype=float)
        for var in lab_vars:
            x_full = group[var].to_numpy(dtype=float)
            mask = ~np.isnan(x_full)
            x = x_full[mask]
            t = t_vals[mask]
            if x.size < 3:
                continue

            # Distribution
            per_var[var].setdefault("mean", []).append(float(tsf.mean(x)))
            per_var[var].setdefault("std", []).append(float(tsf.std(x)))
            per_var[var].setdefault("iqr", []).append(float(tsf.iqr(x)))
            per_var[var].setdefault("skewness", []).append(float(tsf.skewness(x)))
            per_var[var].setdefault("outlier_frac", []).append(float(tsf.outlier_frac(x)))
            per_var[var].setdefault("value_range", []).append(float(tsf.value_range(x)))

            # Autocorrelation
            per_var[var].setdefault("acf_lag1", []).append(float(tsf.acf_lag1(x)))
            per_var[var].setdefault("acf_lag2", []).append(float(tsf.acf_lag2(x)))
            per_var[var].setdefault("acf_lag3", []).append(float(tsf.acf_lag3(x)))
            per_var[var].setdefault("acf_first_zero", []).append(float(tsf.acf_first_zero(x)))
            per_var[var].setdefault("dominant_freq", []).append(float(tsf.dominant_freq(x)))
            per_var[var].setdefault("spectral_entropy", []).append(float(tsf.spectral_entropy(x)))

            # Stationarity
            per_var[var].setdefault("statav", []).append(float(tsf.statav(x)))
            per_var[var].setdefault("std_first_diff", []).append(float(tsf.std_first_difference(x)))

            # Entropy / complexity
            per_var[var].setdefault("perm_entropy_m3", []).append(float(tsf.permutation_entropy_m3(x)))
            per_var[var].setdefault("lz_complexity", []).append(float(tsf.lempel_ziv_complexity(x)))

            # Scaling
            per_var[var].setdefault("dfa_alpha", []).append(float(tsf.dfa_alpha(x)))
            per_var[var].setdefault("spectral_slope", []).append(float(tsf.spectral_slope(x)))

            # Model fit
            per_var[var].setdefault("ar_coef_1", []).append(float(tsf.ar_coef_1(x)))
            per_var[var].setdefault("exp_smooth_alpha", []).append(float(tsf.exp_smoothing_alpha(x)))
            per_var[var].setdefault("turning_point_rate", []).append(float(tsf.turning_point_rate(x)))

            # Trend
            per_var[var].setdefault("spearman_trend", []).append(
                float(tsf.spearman_trend(x, t))
            )

            # Volatility
            per_var[var].setdefault("mean_crossing_rate", []).append(
                float(tsf.mean_crossing_rate(x))
            )
            per_var[var].setdefault("mean_abs_diff", []).append(
                float(tsf.mean_abs_diff(x))
            )

    cross: Dict[str, Dict[str, List[float]]] = {}
    n_bio = len(lab_vars)
    if n_bio >= 2:
        for i in range(n_bio):
            for j in range(i + 1, n_bio):
                b1, b2 = lab_vars[i], lab_vars[j]
                pair = f"{b1}__{b2}"
                cross[pair] = {"level_corr": [], "diff_corr": [], "ccf_lag1": []}
                for _, group in grouped:
                    group = group.sort_values(time_col)
                    sub = group[[b1, b2]].dropna()
                    v1 = sub[b1].to_numpy(dtype=float)
                    v2 = sub[b2].to_numpy(dtype=float)
                    if v1.size < 2:
                        continue
                    cross[pair]["level_corr"].append(float(tsf.level_corr(v1, v2)))
                    # First-difference correlation
                    if v1.size >= 2:
                        cross[pair]["diff_corr"].append(float(tsf.diff_corr(v1, v2)))
                        cross[pair]["ccf_lag1"].append(float(tsf.ccf_lag1(v1, v2)))

    return per_var, cross


def compute_population_stats_from_df(
    df: pd.DataFrame,
    time_col: str = "t",
    id_col: str = "id",
    agg: str = "mean",
    ar_order: int = 3,
    min_patients: int = 3,
) -> pd.DataFrame:
    """Compute pooled per-trajectory statistics from a long-format DataFrame.

    For each trajectory (id), we compute a rich set of time-series statistics
    for each variable and variable pair, then pool across trajectories:

      - value   = mean over trajectories of per-trajectory stat
      - iqr_obs = IQR over trajectories of per-trajectory stat

    Returns:
        DataFrame with columns:
          variable, category, statistic, value, iqr_obs
    """
    lab_vars = [c for c in df.columns if c not in [id_col, time_col]]
    per_var, cross = collect_per_trajectory_stats(
        df=df,
        lab_vars=lab_vars,
        id_col=id_col,
        time_col=time_col,
    )

    rows: List[Dict[str, object]] = []

    # Per-variable stats
    for var in sorted(per_var):
        stats_dict = per_var[var]
        for stat, values in stats_dict.items():
            obs_val = _nanmean(values)
            iqr_val = _iqr(values)
            rows.append(
                {
                    "variable": var,
                    "category": _CATEGORY.get(stat, "unknown"),
                    "statistic": stat,
                    "value": _round_value(stat, obs_val),
                    "iqr_obs": round(iqr_val, 6) if not np.isnan(iqr_val) else np.nan,
                }
            )

    # Cross-variable stats
    for pair in sorted(cross):
        stats_dict = cross[pair]
        for stat in ("level_corr", "diff_corr", "ccf_lag1"):
            values = stats_dict.get(stat, [])
            obs_val = _nanmean(values)
            iqr_val = _iqr(values)
            rows.append(
                {
                    "variable": pair,
                    "category": "cross_variable",
                    "statistic": stat,
                    "value": _round_value(stat, obs_val),
                    "iqr_obs": round(iqr_val, 6) if not np.isnan(iqr_val) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variable", "category", "statistic", "value", "iqr_obs"])

    return pd.DataFrame(rows)


def _infer_dataset_name(data_path: str, dataset_name: Optional[str]) -> str:
    """Infer dataset name from file path if not given."""
    if dataset_name is not None:
        return dataset_name
    base = os.path.basename(data_path)
    name, _ = os.path.splitext(base)
    return name


def save_observed_stats(
    data_path: str,
    out_root: str = "stats",
    dataset_name: Optional[str] = None,
    time_col: str = "t",
    id_col: str = "id",
    agg: str = "mean",
    ar_order: int = 3,
    min_patients: int = 3,
    split: Optional[str] = None,
) -> Tuple[str, str]:
    """Compute and save observed dataset statistics to CSV and JSON.

    Args:
        split: Optional split identifier (e.g. "train" or "test"). When provided,
            output files are named ``ts_stats_<split>.csv`` / ``ts_stats_<split>.json``.

    Returns the CSV and JSON file paths.
    """
    df = pd.read_csv(data_path)
    dataset = _infer_dataset_name(data_path, dataset_name)
    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    stats_df = compute_population_stats_from_df(
        df=df,
        time_col=time_col,
        id_col=id_col,
        agg=agg,
        ar_order=ar_order,
        min_patients=min_patients,
    )

    stem = f"ts_stats_{split}" if split else "ts_stats"
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    json_path = os.path.join(out_dir, f"{stem}.json")

    stats_df.to_csv(csv_path, index=False, na_rep="NaN")
    stats_df.to_json(json_path, orient="records")

    return csv_path, json_path


__all__ = [
    "compute_distribution",
    "compute_autocorrelation",
    "compute_stationarity",
    "permutation_entropy",
    "lempel_ziv_complexity",
    "compute_entropy",
    "compute_nonlinear",
    "dfa_exponent",
    "compute_scaling",
    "compute_model_fit",
    "compute_trend",
    "compute_pairwise_correlations",
    "compute_all_stats",
    "build_population_trajectory",
    "compute_population_stats_from_df",
    "save_observed_stats",
]
