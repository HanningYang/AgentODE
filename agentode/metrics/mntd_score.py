"""Mean Normalized Trajectory Discrepancy (MNTD) metric for ODE systems.

MNTD compares per-trajectory time-series statistics between observed and
synthetic data using data-driven IQR normalisation:

  - For each (variable, statistic) on the observed data:
      obs_value = mean over trajectories of per-trajectory stat
      iqr_obs   = IQR over trajectories of per-trajectory stat
  - For synthetic trajectories:
      syn_value = mean over trajectories of per-trajectory stat
  - Normalised discrepancy:
      d_i  = |obs_value - syn_value|
      d_i' = d_i / iqr_obs  (NaN if iqr_obs ~ 0 or undefined)
  - Aggregate score = mean of all non-NaN d_i' (lower is better).

Observed pooled stats (value + iqr_obs) are precomputed once per problem by
`agentode.agent.time_series_stats.save_observed_stats` and stored in:

    workspace/{problem}/stats/ts_stats.csv
"""

from __future__ import annotations

from typing import Callable, Dict

import os

import numpy as np
import pandas as pd

from agentode.agent.time_series_stats import collect_per_trajectory_stats, save_observed_stats
from agentode.ode import initial_condition_utils


N_PATIENTS_MNTD = 1000


_DECIMALS: Dict[str, int] = {
    "mean":               3,
    "std":                3,
    "iqr":                3,
    "skewness":           3,
    "outlier_frac":       4,
    "acf_lag1":           4,
    "acf_first_zero":     2,
    "dominant_freq":      6,
    "spectral_entropy":   4,
    "statav":             4,
    "std_first_diff":     3,
    "perm_entropy_m3":    4,
    "lz_complexity":      4,
    "dfa_alpha":          4,
    "spectral_slope":     4,
    "ar_coef_1":          4,
    "exp_smooth_alpha":   4,
    "turning_point_rate": 4,
    "spearman_trend":     4,
    "mean_crossing_rate": 4,
    "mean_abs_diff":      3,
    "level_corr":         4,
    "diff_corr":          4,
}
_DEFAULT_DECIMALS = 4


def _round(statistic: str, val: float) -> float:
    if np.isnan(val):
        return np.nan
    dp = _DECIMALS.get(statistic, _DEFAULT_DECIMALS)
    return round(val, dp)


def _nanmean(values) -> float:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def _ensure_ts_stats(problem_name: str) -> pd.DataFrame:
    """Ensure pooled observed stats exist for a problem and return them.

    If `workspace/{problem}/stats/ts_stats.csv` does not exist or lacks
    `iqr_obs`, recompute using `save_observed_stats`.
    """
    out_root = os.path.join("workspace", problem_name, "stats")
    csv_path = os.path.join(out_root, "ts_stats.csv")

    if not os.path.exists(csv_path):
        data_path = os.path.join("data", problem_name, f"{problem_name}.csv")
        save_observed_stats(
            data_path=data_path,
            out_root=out_root,
            dataset_name="",
            time_col="t",
            id_col="id",
            agg="mean",
            ar_order=3,
            min_patients=3,
        )

    df = pd.read_csv(csv_path)

    if "iqr_obs" not in df.columns:
        # Regenerate with the new pooled-per-trajectory semantics.
        data_path = os.path.join("data", problem_name, f"{problem_name}.csv")
        save_observed_stats(
            data_path=data_path,
            out_root=out_root,
            dataset_name="",
            time_col="t",
            id_col="id",
            agg="mean",
            ar_order=3,
            min_patients=3,
        )
        df = pd.read_csv(csv_path)

    return df


def _mntd_distance_from_trajectories(
    problem_name: str,
    pooled_obs: pd.DataFrame,
    synthetic_traj: np.ndarray,
    t_eval: np.ndarray,
    biomarker_names: list[str],
    verbose: bool = False,
) -> float | None:
    """Compute MNTD between pooled observed stats and synthetic trajectories."""
    n_patients, n_times, n_bio = synthetic_traj.shape
    assert n_bio == len(biomarker_names)

    # Build synthetic long-format DataFrame: id, t, biomarkers.
    records = []
    for pid in range(n_patients):
        for t_idx, t in enumerate(t_eval):
            row = {"id": pid, "t": float(t)}
            for b_idx, name in enumerate(biomarker_names):
                row[name] = float(synthetic_traj[pid, t_idx, b_idx])
            records.append(row)
    synthetic_df = pd.DataFrame.from_records(records)

    syn_per_var, syn_cross = collect_per_trajectory_stats(
        df=synthetic_df,
        lab_vars=biomarker_names,
        id_col="id",
        time_col="t",
    )

    rows = []

    # Per-variable stats
    for var in biomarker_names:
        syn_v = syn_per_var.get(var, {})
        obs_rows = pooled_obs[pooled_obs["variable"] == var]
        for _, orow in obs_rows.iterrows():
            stat = str(orow["statistic"])
            obs_val = float(orow["value"])
            iqr_val = float(orow.get("iqr_obs", np.nan))
            syn_vals = syn_v.get(stat, [])
            syn_val = _nanmean(syn_vals)

            if np.isnan(obs_val) or np.isnan(syn_val):
                norm_diff = np.nan
            else:
                raw_diff = abs(obs_val - syn_val)
                raw_diff_rounded = _round(stat, raw_diff)
                if raw_diff_rounded == 0.0:
                    norm_diff = 0.0
                elif np.isnan(iqr_val) or iqr_val < 1e-6:
                    norm_diff = np.nan
                else:
                    norm_diff = raw_diff / iqr_val

            rows.append(
                {
                    "variable": var,
                    "statistic": stat,
                    "normalized_diff": norm_diff,
                }
            )

    # Cross-variable stats
    obs_cross = pooled_obs[pooled_obs["category"] == "cross_variable"]
    for pair, group in obs_cross.groupby("variable"):
        syn_c = syn_cross.get(pair, {})
        for _, orow in group.iterrows():
            stat = str(orow["statistic"])
            obs_val = float(orow["value"])
            iqr_val = float(orow.get("iqr_obs", np.nan))
            syn_vals = syn_c.get(stat, [])
            syn_val = _nanmean(syn_vals)

            if np.isnan(obs_val) or np.isnan(syn_val):
                norm_diff = np.nan
            else:
                raw_diff = abs(obs_val - syn_val)
                raw_diff_rounded = _round(stat, raw_diff)
                if raw_diff_rounded == 0.0:
                    norm_diff = 0.0
                elif np.isnan(iqr_val) or iqr_val < 1e-6:
                    norm_diff = np.nan
                else:
                    norm_diff = raw_diff / iqr_val

            rows.append(
                {
                    "variable": pair,
                    "statistic": stat,
                    "normalized_diff": norm_diff,
                }
            )

    if not rows:
        return None

    result = pd.DataFrame(rows)
    valid_norm = result["normalized_diff"].dropna()
    if valid_norm.empty:
        return None

    mntd = float(valid_norm.mean())
    if verbose:
        print(f"[MNTD] Mean normalized trajectory discrepancy: {mntd:.6f}")
    return mntd


def evaluate_system_mntd(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    standardization: bool = True,  # kept for API compatibility; unused here
) -> float | None:
    """Evaluate a system via Mean Normalized Trajectory Discrepancy (MNTD)."""
    from agentode.ode import ode_simulator
    from agentode.core import param_utils

    if param_distributions is None:
        return None

    # Load IC config and biomarker/time metadata.
    config = initial_condition_utils.load_ic_config(problem_name)
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    all_biomarker_names = initial_condition_utils.get_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    # Ensure pooled observed stats exist and load them.
    pooled_obs = _ensure_ts_stats(problem_name)

    def param_sampler(n: int) -> np.ndarray:
        return param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n,
            distribution="lognormal",
        )

    def _filter_valid(trajectories: np.ndarray) -> np.ndarray:
        valid_mask, _ = ode_simulator.check_trajectory_normal_range_validity(
            trajectories,
            config=config,
            check_nans=True,
        )
        observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]
        return trajectories[valid_mask][..., observed_indices]

    if backend == "gpu":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def simulator(n_patients: int, sampler_fn, t_grid: np.ndarray) -> np.ndarray:
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config,
                sample_size=n_patients,
                random_seed=config.get("random_seed", None),
            )
            traj_t = ode_simulator.simulate_ode_system_torch(
                system_func=system_func,
                initial_conditions=ic_array,
                time_grid=t_grid,
                params=sampler_fn(n_patients),
                device=device,
                dtype=torch.float32,
            )
            return _filter_valid(traj_t.detach().cpu().numpy())

    else:

        def simulator(n_patients: int, sampler_fn, t_grid: np.ndarray) -> np.ndarray:
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config,
                sample_size=n_patients,
                random_seed=config.get("random_seed", None),
            )
            trajectories = ode_simulator.simulate_ode_system(
                system_func=system_func,
                initial_conditions=ic_array,
                time_grid=t_grid,
                params=sampler_fn(n_patients),
                method="odeint",
            )
            return _filter_valid(trajectories)

    trajectories = simulator(N_PATIENTS_MNTD, param_sampler, t_eval)

    if trajectories.size == 0:
        return None

    distance = _mntd_distance_from_trajectories(
        problem_name=problem_name,
        pooled_obs=pooled_obs,
        synthetic_traj=trajectories,
        t_eval=t_eval,
        biomarker_names=biomarker_names,
        verbose=verbose,
    )

    if distance is None or not np.isfinite(distance):
        return None

    return distance


def compare_stats_iqr_from_dfs(
    obs_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    out_path: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """Utility to compare stats between two DataFrames using IQR normalisation.

    This mirrors the MNTD logic but works on arbitrary observed/synthetic
    DataFrames (columns: id, t, biomarkers) without going through the ODE
    simulator. It is intended for offline analysis and debugging.
    """
    lab_vars = [c for c in obs_df.columns if c not in ["id", "t"]]

    # Pooled observed stats
    obs_per_var, obs_cross = collect_per_trajectory_stats(
        df=obs_df,
        lab_vars=lab_vars,
        id_col="id",
        time_col="t",
    )

    rows = []
    # Build pooled obs table in the same shape as ts_stats.csv
    for var in sorted(obs_per_var):
        stats_dict = obs_per_var[var]
        for stat, values in stats_dict.items():
            obs_val = _nanmean(values)
            iqr_val = np.percentile(values, 75) - np.percentile(values, 25)
            rows.append(
                {
                    "variable": var,
                    "category": "unknown",
                    "statistic": stat,
                    "value": _round(stat, obs_val),
                    "iqr_obs": round(float(iqr_val), 6),
                }
            )
    for pair in sorted(obs_cross):
        stats_dict = obs_cross[pair]
        for stat, values in stats_dict.items():
            obs_val = _nanmean(values)
            iqr_val = np.percentile(values, 75) - np.percentile(values, 25)
            rows.append(
                {
                    "variable": pair,
                    "category": "cross_variable",
                    "statistic": stat,
                    "value": _round(stat, obs_val),
                    "iqr_obs": round(float(iqr_val), 6),
                }
            )

    pooled_obs = pd.DataFrame(rows)

    # Synthetic pooled stats
    syn_per_var, syn_cross = collect_per_trajectory_stats(
        df=syn_df,
        lab_vars=lab_vars,
        id_col="id",
        time_col="t",
    )

    comp_rows = []
    # Per-variable
    for var in lab_vars:
        syn_v = syn_per_var.get(var, {})
        obs_rows = pooled_obs[pooled_obs["variable"] == var]
        for _, orow in obs_rows.iterrows():
            stat = str(orow["statistic"])
            obs_val = float(orow["value"])
            iqr_val = float(orow.get("iqr_obs", np.nan))
            syn_vals = syn_v.get(stat, [])
            syn_val = _nanmean(syn_vals)

            if np.isnan(obs_val) or np.isnan(syn_val):
                norm_diff = np.nan
                raw_diff_rounded = np.nan
            else:
                raw_diff = abs(obs_val - syn_val)
                raw_diff_rounded = _round(stat, raw_diff)
                if raw_diff_rounded == 0.0:
                    norm_diff = 0.0
                elif np.isnan(iqr_val) or iqr_val < 1e-6:
                    norm_diff = np.nan
                else:
                    norm_diff = raw_diff / iqr_val

            comp_rows.append(
                {
                    "variable": var,
                    "category": "unknown",
                    "statistic": stat,
                    "obs_value": _round(stat, obs_val),
                    "syn_value": _round(stat, syn_val),
                    "raw_difference": raw_diff_rounded,
                    "iqr_obs": round(iqr_val, 6) if not np.isnan(iqr_val) else np.nan,
                    "normalized_diff": round(norm_diff, 6) if not np.isnan(norm_diff) else np.nan,
                }
            )

    # Cross-variable
    for pair, group in pooled_obs[pooled_obs["category"] == "cross_variable"].groupby("variable"):
        syn_c = syn_cross.get(pair, {})
        for _, orow in group.iterrows():
            stat = str(orow["statistic"])
            obs_val = float(orow["value"])
            iqr_val = float(orow.get("iqr_obs", np.nan))
            syn_vals = syn_c.get(stat, [])
            syn_val = _nanmean(syn_vals)

            if np.isnan(obs_val) or np.isnan(syn_val):
                norm_diff = np.nan
                raw_diff_rounded = np.nan
            else:
                raw_diff = abs(obs_val - syn_val)
                raw_diff_rounded = _round(stat, raw_diff)
                if raw_diff_rounded == 0.0:
                    norm_diff = 0.0
                elif np.isnan(iqr_val) or iqr_val < 1e-6:
                    norm_diff = np.nan
                else:
                    norm_diff = raw_diff / iqr_val

            comp_rows.append(
                {
                    "variable": pair,
                    "category": "cross_variable",
                    "statistic": stat,
                    "obs_value": _round(stat, obs_val),
                    "syn_value": _round(stat, syn_val),
                    "raw_difference": raw_diff_rounded,
                    "iqr_obs": round(iqr_val, 6) if not np.isnan(iqr_val) else np.nan,
                    "normalized_diff": round(norm_diff, 6) if not np.isnan(norm_diff) else np.nan,
                }
            )

    result = pd.DataFrame(comp_rows)
    valid_norm = result["normalized_diff"].dropna()
    agg_score = float(valid_norm.mean()) if len(valid_norm) > 0 else np.nan

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        result.to_csv(out_path, index=False, na_rep="NaN")

    return result, agg_score
