"""Shared summary-statistics utilities for ODE evaluation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from llmode.ode import initial_condition_utils


def polynomial_quantile_stats_deg01(values: np.ndarray) -> np.ndarray:
    """Fit linear polynomial (deg0, deg1) to empirical quantile function.

    Assumes `values` are already standardized.
    """
    values = values[~np.isnan(values)]
    if values.size < 2:
        return np.array([0.0, 0.0], dtype=float)

    sorted_vals = np.sort(values)
    n = sorted_vals.size
    quantiles = np.arange(1, n + 1, dtype=float) / (n + 1)

    coeffs = np.polyfit(quantiles, sorted_vals, deg=1)
    # Return [deg0, deg1] for convenience.
    return np.array([coeffs[1], coeffs[0]], dtype=float)


def compute_standardization_params(
    df: pd.DataFrame,
    biomarker_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute mean/std from observed data for each biomarker column."""
    params: Dict[str, Dict[str, float]] = {}
    for col in biomarker_cols:
        values = df[col].dropna().to_numpy()
        mean = float(np.mean(values)) if values.size > 0 else 0.0
        std = float(np.std(values)) if values.size > 1 else 1.0
        if std < 1e-10:
            std = 1.0
        params[col] = {"mean": mean, "std": std}
    return params


def standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize values using given mean and std."""
    return (values - mean) / std


def compute_summary_stats(
    trajectories: np.ndarray,
    t_eval: np.ndarray,
    biomarker_names: List[str],
    std_params: Dict[str, Dict[str, float]],
    standardization: bool = True,
) -> np.ndarray:
    """Compute summary statistics from simulated trajectories.

    Stats per biomarker:
      - 2: Quantile polynomial deg0, deg1 on standardized values
      - 1: Lag-1 autocorrelation (across time)
      - 1: Population-level trend (Spearman correlation between time and
           population mean trajectory)

    Cross-biomarker stats:
      - For each pair (b1, b2) with b2 > b1: correlation of first differences.
    """
    n_patients, n_times, n_bio = trajectories.shape
    assert n_bio == len(biomarker_names)

    stats: List[float] = []

    # 1. Marginal quantile polynomials (2 * n_bio).
    #    When `standardization=True`, coefficients are computed on standardized
    #    values using `std_params`; otherwise raw values are used.
    for b, bio_name in enumerate(biomarker_names):
        all_values = trajectories[:, :, b].ravel()
        all_values = all_values[~np.isnan(all_values)]
        if standardization:
            values_for_quantile = standardize(
                all_values,
                std_params[bio_name]["mean"],
                std_params[bio_name]["std"],
            )
        else:
            values_for_quantile = all_values
        coeffs = polynomial_quantile_stats_deg01(values_for_quantile)
        stats.extend(coeffs.tolist())

    # 2. Temporal: lag-1 autocorrelation (n_bio).
    for b, _bio_name in enumerate(biomarker_names):
        y_t: List[float] = []
        y_t1: List[float] = []
        for i in range(n_patients):
            vals = trajectories[i, :, b]
            if not np.any(np.isnan(vals)) and vals.size > 1:
                y_t.extend(vals[1:].tolist())
                y_t1.extend(vals[:-1].tolist())

        if len(y_t) > 1:
            autocorr, _ = spearmanr(y_t, y_t1)
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        stats.append(autocorr)

    # 3. Population-level trend: Spearman correlation between time and
    #    population mean trajectory for each biomarker (n_bio).
    for b, _bio_name in enumerate(biomarker_names):
        mean_traj: List[float] = []
        for t_idx in range(n_times):
            vals = trajectories[:, t_idx, b]
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                mean_traj.append(float(np.mean(vals)))
            else:
                mean_traj.append(np.nan)

        mean_traj_arr = np.asarray(mean_traj, dtype=float)
        time_vec = np.asarray(t_eval, dtype=float)
        mask = ~np.isnan(mean_traj_arr)
        mean_traj_masked = mean_traj_arr[mask]
        time_vec_masked = time_vec[mask]

        if mean_traj_masked.size > 1:
            trend, _ = spearmanr(time_vec_masked, mean_traj_masked)
            if np.isnan(trend):
                trend = 0.0
        else:
            trend = 0.0
        stats.append(float(trend))

    # 4. Dynamic: difference correlations between biomarker pairs.
    for b1 in range(n_bio):
        for b2 in range(b1 + 1, n_bio):
            diffs1: List[float] = []
            diffs2: List[float] = []
            for i in range(n_patients):
                v1 = trajectories[i, :, b1]
                v2 = trajectories[i, :, b2]
                if (
                    not np.any(np.isnan(v1))
                    and not np.any(np.isnan(v2))
                    and v1.size > 1
                    and v2.size > 1
                ):
                    dv1 = np.diff(v1)
                    dv2 = np.diff(v2)
                    m = min(dv1.size, dv2.size)
                    diffs1.extend(dv1[:m].tolist())
                    diffs2.extend(dv2[:m].tolist())

            if len(diffs1) > 1:
                corr, _ = spearmanr(diffs1, diffs2)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            stats.append(corr)

    return np.asarray(stats, dtype=float)


def compute_summary_stats_from_df(
    df: pd.DataFrame,
    biomarker_cols: List[str],
    std_params: Dict[str, Dict[str, float]],
    subject_col: str = "id",
    time_col: str = "t",
    verbose: bool = False,
    standardization: bool = True,
) -> np.ndarray:
    """Compute summary statistics from observed longitudinal data."""
    stats: List[float] = []

    if verbose:
        print("\n" + "=" * 80)
        print("COMPUTING SUMMARY STATISTICS FROM OBSERVED DATA")
        print("=" * 80)
        print(f"\nDataset shape: {df.shape}")
        print(f"Unique subjects: {df[subject_col].nunique()}")
        print(f"\nBiomarker columns: {biomarker_cols}")

    # 1. Marginal quantile polynomials.
    #    When `standardization=True`, coefficients are computed on standardized
    #    values using `std_params`; otherwise raw values are used.
    if verbose:
        print("\n[1] Quantile Polynomial Coefficients (deg0, deg1):")
    for col in biomarker_cols:
        values = df[col].dropna().to_numpy()
        if verbose:
            print(f"\n  {col}:")
            print(
                f"    Raw - Count: {len(values)}, Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}"
            )
            print(
                f"    Standardization - Mean: {std_params[col]['mean']:.4f}, Std: {std_params[col]['std']:.4f}"
            )
        if standardization:
            values_for_quantile = standardize(
                values,
                std_params[col]["mean"],
                std_params[col]["std"],
            )
        else:
            values_for_quantile = values
        coeffs = polynomial_quantile_stats_deg01(values_for_quantile)
        if verbose:
            print(f"    Quantile poly deg0: {coeffs[0]:.6f}, deg1: {coeffs[1]:.6f}")
        stats.extend(coeffs.tolist())

    # 2. Temporal: lag-1 autocorrelation per biomarker across episodes.
    if verbose:
        print("\n[2] Lag-1 Autocorrelation:")
    grouped = df.groupby(subject_col)
    for col in biomarker_cols:
        y_t: List[float] = []
        y_t1: List[float] = []
        for _, group in grouped:
            g = group.sort_values(time_col)
            vals = g[col].dropna().to_numpy()
            if vals.size > 1:
                y_t.extend(vals[1:].tolist())
                y_t1.extend(vals[:-1].tolist())

        if len(y_t) > 1:
            autocorr, _ = spearmanr(y_t, y_t1)
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        if verbose:
            print(f"  {col}: {autocorr:.6f} (from {len(y_t)} lag pairs)")
        stats.append(autocorr)

    # 3. Population-level trend: Spearman correlation between time and
    #    population mean trajectory for each biomarker.
    if verbose:
        print("\n[3] Population-level trend (Spearman time vs mean biomarker):")
    for col in biomarker_cols:
        sub = df[[time_col, col]].dropna(subset=[col])
        if sub.empty:
            trend = 0.0
            n_timepoints = 0
        else:
            grouped_time = sub.groupby(time_col)[col].mean()
            time_vec = grouped_time.index.to_numpy(dtype=float)
            mean_traj = grouped_time.to_numpy(dtype=float)
            n_timepoints = mean_traj.size
            if n_timepoints > 1:
                trend, _ = spearmanr(time_vec, mean_traj)
                if np.isnan(trend):
                    trend = 0.0
            else:
                trend = 0.0
        if verbose:
            print(f"  {col}: {trend:.6f} (from {n_timepoints} time points)")
        stats.append(float(trend))

    # 4. Dynamic: difference correlations between biomarker pairs.
    if verbose:
        print("\n[4] Difference Correlations (between biomarker pairs):")
    for i, col1 in enumerate(biomarker_cols):
        for j, col2 in enumerate(biomarker_cols):
            if j <= i:
                continue
            diffs1: List[float] = []
            diffs2: List[float] = []
            for _, group in grouped:
                g = group.sort_values(time_col)
                # v1 = g[col1].dropna().to_numpy()
                # v2 = g[col2].dropna().to_numpy()
                sub = g[[col1, col2]].dropna()  
                v1 = sub[col1].to_numpy()
                v2 = sub[col2].to_numpy()
                m = min(v1.size, v2.size)
                if m > 1:
                    dv1 = np.diff(v1[:m])
                    dv2 = np.diff(v2[:m])
                    diffs1.extend(dv1.tolist())
                    diffs2.extend(dv2.tolist())

            if len(diffs1) > 1:
                corr, _ = spearmanr(diffs1, diffs2)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            if verbose:
                print(f"  {col1} vs {col2}: {corr:.6f} (from {len(diffs1)} diff pairs)")
            stats.append(corr)

    result = np.asarray(stats, dtype=float)

    if verbose:
        stat_names = get_stat_names(biomarker_cols)
        print("\n" + "-" * 80)
        print(f"SUMMARY STATISTICS VECTOR (s_obs) - Total: {len(result)}")
        print("-" * 80)
        print(f"{'Index':<8} {'Statistic Name':<40} {'Value':>12}")
        print("-" * 80)
        for idx, (name, value) in enumerate(zip(stat_names, result)):
            print(f"{idx:<8} {name:<40} {value:12.6f}")
        print("=" * 80 + "\n")

    return result


def get_stat_names(biomarker_cols: List[str]) -> List[str]:
    """Return human-readable names for summary statistics."""
    names: List[str] = []

    # Quantile polynomial coefficients.
    for col in biomarker_cols:
        names.append(f"{col}_std_quantile_deg0")
        names.append(f"{col}_std_quantile_deg1")

    # Lag-1 autocorrelation.
    for col in biomarker_cols:
        names.append(f"{col}_lag1_autocorr")

    # Population-level trend.
    for col in biomarker_cols:
        names.append(f"{col}_pop_trend_spearman")

    # Difference correlations.
    for i, col1 in enumerate(biomarker_cols):
        for j, col2 in enumerate(biomarker_cols):
            if j <= i:
                continue
            names.append(f"diff_corr_{col1[:2]}_{col2[:2]}")

    return names


_OBSERVED_CACHE: Dict[Tuple[str, bool], Dict[str, object]] = {}


def get_observed_summary(problem_name: str, standardization: bool = True) -> Dict[str, object]:
    """Load and cache observed-data summaries for a problem."""
    cache_key = (problem_name, standardization)
    if cache_key in _OBSERVED_CACHE:
        return _OBSERVED_CACHE[cache_key]

    config = initial_condition_utils.load_ic_config(problem_name)
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    all_biomarker_names = initial_condition_utils.get_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    observed_data_path = f"data/{problem_name}/{problem_name}.csv"
    observed_df = pd.read_csv(observed_data_path)

    std_params = compute_standardization_params(observed_df, biomarker_names)
    s_obs = compute_summary_stats_from_df(
        df=observed_df,
        biomarker_cols=biomarker_names,
        std_params=std_params,
        subject_col="id",
        time_col="t",
        verbose=False,
        standardization=standardization,
    )
    stat_names = get_stat_names(biomarker_names)

    payload: Dict[str, object] = {
        "config": config,
        "biomarker_names": biomarker_names,
        "all_biomarker_names": all_biomarker_names,
        "t_eval": t_eval,
        "observed_df": observed_df,
        "std_params": std_params,
        "s_obs": s_obs,
        "stat_names": stat_names,
    }
    _OBSERVED_CACHE[cache_key] = payload
    return payload
