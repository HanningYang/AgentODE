"""Summary-statistics distance for ODE systems.

Historically this computed a simple Euclidean distance in summary-stat space:

    distance = ||s_obs - mu_hat||_2

where `s_obs` are observed summary statistics and `mu_hat` is the mean
summary-statistics vector from simulations.

We now use a more interpretable time-series-statistics comparison that matches
the logic in `analysis/posthoc/compare_stats.py`:

  - Compute population time-series stats (distribution, autocorrelation,
    entropy, etc.) for both observed and synthetic trajectories.
  - Restrict to the subset of stats exposed as LLM tools.
  - Apply per-stat normalisation rules (range / absolute / relative).
  - Aggregate the mean normalised absolute error across all stats.

This preserves the public API while changing the underlying distance metric.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from llmode.metrics.summary_stats import get_observed_summary
from llmode.agent.time_series_stats import compute_population_stats_from_df


N_PATIENTS_EUCLIDEAN = 1000


# ---------------------------------------------------------------------------
# Stat definitions & normalisation (mirrors analysis/posthoc/compare_stats.py)
# ---------------------------------------------------------------------------

_TOOL_STATS: set[tuple[str, str]] = {
    ("distribution", "mean"),
    ("distribution", "variance"),
    ("distribution", "skewness"),
    ("autocorrelation", "acf_lag1"),
    ("autocorrelation", "acf_lag3"),
    ("autocorrelation", "acf_1e_timescale"),
    ("autocorrelation", "dominant_freq"),
    ("stationarity", "statav"),
    ("stationarity", "std_first_diff"),
    ("entropy", "perm_entropy_m3"),
    ("nonlinear", "time_reversal_asym"),
    ("model_fit", "ar_coef_1"),
    ("model_fit", "ar_coef_2"),
    ("model_fit", "turning_point_rate"),
    ("trend", "spearman_trend"),
    ("cross_variable", "diff_corr"),
}

# Per-stat normalization rules:
#   "range"    → divide by a fixed range constant
#   "absolute" → absolute difference only (no division)
#   "relative" → divide by max(|real|, eps)
_STAT_NORM: dict[str, tuple[str, float]] = {
    # (mode, range_constant)  — range_constant only used for "range" mode
    "acf_lag1":           ("range",    2.0),
    "acf_lag3":           ("range",    2.0),
    "perm_entropy_m3":    ("range",    1.792),   # ln(6)
    "ar_coef_1":          ("absolute", 0.0),
    "ar_coef_2":          ("absolute", 0.0),
    "time_reversal_asym": ("absolute", 0.0),
    "skewness":           ("absolute", 0.0),
    "turning_point_rate": ("absolute", 0.0),
    "mean":               ("relative", 0.0),
    "variance":           ("relative", 0.0),
    "std_first_diff":     ("relative", 0.0),
    "statav":             ("relative", 0.0),
    "spearman_trend":     ("relative", 0.0),
    "diff_corr":          ("relative", 0.0),
    "dominant_freq":      ("relative", 0.0),
    "acf_1e_timescale":   ("relative", 0.0),
}


def _filter_to_tool_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a ts-stats DataFrame to the subset used as tools."""
    mask = df.apply(
        lambda row: (row["category"], row["statistic"]) in _TOOL_STATS, axis=1
    )
    return df[mask].copy()


def _compute_normalised_error(
    stat_name: str,
    real: float,
    synth: float,
    eps: float,
) -> float:
    """Apply the per-stat normalization rule and return the error for one stat."""
    abs_diff = abs(real - synth)
    mode, range_const = _STAT_NORM.get(stat_name, ("relative", 0.0))
    if mode == "range":
        return abs_diff / range_const
    elif mode == "absolute":
        return abs_diff
    else:  # relative
        return abs_diff / max(abs(real), eps)


def _ts_stats_distance_from_trajectories(
    observed_df: pd.DataFrame,
    synthetic_traj: np.ndarray,
    t_eval: np.ndarray,
    biomarker_names: list[str],
    eps: float = 0.01,
) -> float | None:
    """Compute mean normalised ts-stats error between observed and synthetic."""
    # Build synthetic long-format DataFrame: id, t, biomarkers.
    n_patients, n_times, n_bio = synthetic_traj.shape
    assert n_bio == len(biomarker_names)

    records = []
    for pid in range(n_patients):
        for t_idx, t in enumerate(t_eval):
            row = {"id": pid, "t": float(t)}
            for b_idx, name in enumerate(biomarker_names):
                row[name] = float(synthetic_traj[pid, t_idx, b_idx])
            records.append(row)
    synthetic_df = pd.DataFrame.from_records(records)

    # Compute population-level ts stats for observed and synthetic data.
    obs_stats = compute_population_stats_from_df(
        observed_df,
        time_col="t",
        id_col="id",
        agg="mean",
        ar_order=3,
        min_patients=1,
    )
    syn_stats = compute_population_stats_from_df(
        synthetic_df,
        time_col="t",
        id_col="id",
        agg="mean",
        ar_order=3,
        min_patients=1,
    )

    if obs_stats.empty or syn_stats.empty:
        return None

    obs = _filter_to_tool_stats(obs_stats)
    syn = _filter_to_tool_stats(syn_stats)

    merged = obs.merge(
        syn,
        on=["variable", "category", "statistic"],
        suffixes=("_real", "_synth"),
    )

    if merged.empty:
        return None

    real_vals = merged["value_real"].astype(float)
    synth_vals = merged["value_synth"].astype(float)

    errors = []
    for stat, r, s in zip(merged["statistic"], real_vals, synth_vals):
        if np.isnan(r) or np.isnan(s):
            errors.append(np.nan)
        else:
            errors.append(_compute_normalised_error(str(stat), float(r), float(s), eps))

    merged["error"] = errors
    valid = merged.dropna(subset=["error"])
    if valid.empty:
        return None

    return float(valid["error"].mean())


def evaluate_system_euclidean_distance(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    standardization: bool = True,
) -> float | None:
    """Evaluate a system via Euclidean distance in summary-stat space."""
    from llmode.ode import initial_condition_utils, ode_simulator
    from llmode.core import param_utils

    if param_distributions is None:
        return None

    obs = get_observed_summary(problem_name, standardization=standardization)
    config = obs["config"]
    biomarker_names = obs["biomarker_names"]
    all_biomarker_names = obs["all_biomarker_names"]
    t_eval = obs["t_eval"]
    observed_df = obs["observed_df"]

    def param_sampler(n: int) -> np.ndarray:
        return param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n,
            distribution="lognormal",
        )

    def _filter_valid(trajectories: np.ndarray) -> np.ndarray:
        valid_mask, _ = ode_simulator.check_trajectory_normal_range_validity(
            trajectories, config=config, check_nans=True,
        )
        observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]
        return trajectories[valid_mask][..., observed_indices]

    if backend == "gpu":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def simulator(n_patients, sampler_fn, t_grid):
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config, sample_size=n_patients,
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
        def simulator(n_patients, sampler_fn, t_grid):
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config, sample_size=n_patients,
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

    trajectories = simulator(N_PATIENTS_EUCLIDEAN, param_sampler, t_eval)

    if trajectories.size == 0:
        return None

    distance = _ts_stats_distance_from_trajectories(
        observed_df=observed_df,
        synthetic_traj=trajectories,
        t_eval=t_eval,
        biomarker_names=biomarker_names,
        eps=0.01,
    )

    if not np.isfinite(distance):
        return None

    if verbose:
        print(f"Mean normalised ts-stats error: {distance:.2f}")

    return distance
