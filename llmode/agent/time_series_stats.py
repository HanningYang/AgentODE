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
    add("distribution", "within_patient_mean", tsf.mean(x))
    add("distribution", "within_patient_variance", tsf.variance(x))
    add("distribution", "within_patient_skewness", tsf.skewness(x))

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
    "within_patient_mean": 3,
    "within_patient_variance": 3,
    "within_patient_skewness": 3,
    "acf_lag1": 4,
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


def build_population_trajectory(
    df: pd.DataFrame,
    variable: str,
    time_col: str = "t",
    agg: str = "mean",
    min_patients: int = 3,
) -> pd.Series:
    """Aggregate individual series into a population-level trajectory."""
    grouped = df.dropna(subset=[variable]).groupby(time_col)[variable]
    counts = grouped.count()
    if agg == "mean":
        traj = grouped.mean()
    elif agg == "median":
        traj = grouped.median()
    elif agg == "std":
        traj = grouped.std()
    elif agg == "count":
        traj = grouped.count()
    else:
        raise ValueError(f"Unknown aggregation: {agg}")
    return traj[counts >= min_patients].sort_index()


def compute_population_stats_from_df(
    df: pd.DataFrame,
    time_col: str = "t",
    id_col: str = "id",
    agg: str = "mean",
    ar_order: int = 3,
    min_patients: int = 3,
) -> pd.DataFrame:
    """Compute population-level statistics from a long-format DataFrame."""
    lab_vars = [c for c in df.columns if c not in [id_col, time_col]]
    all_results: List[pd.DataFrame] = []

    for var in lab_vars:
        traj = build_population_trajectory(
            df,
            variable=var,
            time_col=time_col,
            agg=agg,
            min_patients=min_patients,
        )
        # if len(traj) < 5:
        #     continue

        x = traj.values.astype(float)
        time_index = traj.index.values.astype(float)
        results = compute_all_stats(x, time_index, var_name=var, ar_order=ar_order)
        all_results.append(format_per_variable(results, var))

    if len(lab_vars) >= 2:
        pair_results = compute_pairwise_correlations(df, lab_vars, id_col=id_col, time_col=time_col)
        all_results.append(format_pairwise(pair_results))

    if not all_results:
        return pd.DataFrame(columns=["variable", "category", "statistic", "value"])

    return pd.concat(all_results, ignore_index=True)


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
) -> Tuple[str, str]:
    """Compute and save observed dataset statistics to CSV and JSON.

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

    csv_path = os.path.join(out_dir, "ts_stats.csv")
    json_path = os.path.join(out_dir, "ts_stats.json")

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
