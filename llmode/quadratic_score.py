"""Quadratic summary-statistics score for ODE systems.

Uses the same summary statistics as the synthetic likelihood module, but
replaces the log synthetic likelihood with a simple quadratic score:

    score = -sum((s_obs - mu_hat)**2)

where s_obs are summary statistics from observed data and mu_hat is the
mean summary-statistics vector from simulated trajectories.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from llmode.summary_stats import (
    compute_standardization_params,
    compute_summary_stats,
    compute_summary_stats_from_df,
    get_observed_summary,
    get_stat_names,
)


N_PATIENTS_QUADRATIC = 1000


class _PhysioRejection(Exception):
    """Internal signal that a simulation failed physiological validity."""

    def __init__(self, valid_fraction: float):
        super().__init__(f"valid_fraction={valid_fraction:.3f}")
        self.valid_fraction = float(valid_fraction)


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


def compute_standardization_params(df: pd.DataFrame, biomarker_cols: List[str]) -> Dict[str, Dict[str, float]]:
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


def get_stat_names(biomarker_cols: List[str]) -> List[str]:
    """Return human-readable names for summary statistics."""
    from llmode.summary_stats import get_stat_names as _get_stat_names

    return _get_stat_names(biomarker_cols)


def evaluate_system_quadratic_score(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
) -> float | None:
    """Evaluate a system via quadratic distance in summary-stat space."""
    from llmode import initial_condition_utils
    from llmode import ode_simulator
    from llmode import param_utils

    # If no parameter distributions are available, treat as unevaluable.
    if param_distributions is None:
        return None

    # Load configuration and observed data (cached per problem).
    obs = get_observed_summary(problem_name)
    config = obs["config"]
    biomarker_names = obs["biomarker_names"]
    all_biomarker_names = obs["all_biomarker_names"]
    t_eval = obs["t_eval"]
    std_params = obs["std_params"]
    s_obs = obs["s_obs"]
    stat_names = obs["stat_names"]

    def param_sampler(n: int) -> np.ndarray:
        return param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n,
            distribution="lognormal",
        )

    if backend == "gpu":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def simulator(
            n_patients: int,
            sampler_fn: Callable[[int], np.ndarray],
            t_grid: np.ndarray,
        ) -> np.ndarray:
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config,
                sample_size=n_patients,
                random_seed=config.get("random_seed", None),
            )
            param_sets = sampler_fn(n_patients)

            traj_t = ode_simulator.simulate_ode_system_torch(
                system_func=system_func,
                initial_conditions=ic_array,
                time_grid=t_grid,
                params=param_sets,
                device=device,
                dtype=torch.float32,
            )
            trajectories = traj_t.detach().cpu().numpy()

            valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
                trajectories,
                config=config,
                check_nans=True,
            )
            valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
            if valid_fraction < 0.8:
                if sample_order not in (0, None):
                    raise _PhysioRejection(valid_fraction)

            observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]
            if not np.any(valid_mask):
                empty = trajectories[valid_mask][..., observed_indices]
                return empty
            return trajectories[valid_mask][..., observed_indices]
    else:
        def simulator(
            n_patients: int,
            sampler_fn: Callable[[int], np.ndarray],
            t_grid: np.ndarray,
        ) -> np.ndarray:
            ic_array, _ = initial_condition_utils.generate_initial_conditions(
                config,
                sample_size=n_patients,
                random_seed=config.get("random_seed", None),
            )
            param_sets = sampler_fn(n_patients)
            trajectories = ode_simulator.simulate_ode_system(
                system_func=system_func,
                initial_conditions=ic_array,
                time_grid=t_grid,
                params=param_sets,
                method="odeint",
            )
            # Drop patients whose trajectories are numerically or physiologically invalid.
            valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
                trajectories,
                config=config,
                check_nans=True,
            )
            valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
            if valid_fraction < 0.8:
                # For the very first template/system (sample_order == 0), allow a
                # more permissive evaluation so that we always obtain an initial
                # baseline score, even if many trajectories are invalid.
                # For all later samples, enforce the 0.8 threshold strictly.
                if sample_order not in (0, None):
                    # Signal to the outer evaluator that this system should be rejected.
                    raise _PhysioRejection(valid_fraction)

            # Map observed biomarker names to their indices in the full config
            # order so we can always return trajectories whose biomarker dimension
            # matches `biomarker_names`, even if all trajectories are invalid.
            observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]

            if not np.any(valid_mask):
                # No valid trajectories: return an empty array with the correct
                # biomarker dimension so downstream summary-stat routines do not
                # see a mismatch between n_bio and len(biomarker_names).
                empty = trajectories[valid_mask][..., observed_indices]
                return empty

            # Restrict trajectories to valid patients and observed biomarkers.
            return trajectories[valid_mask][..., observed_indices]

    try:
        # Single simulation with N_PATIENTS_QUADRATIC patients.
        trajectories = simulator(
            N_PATIENTS_QUADRATIC,
            param_sampler,
            t_eval,
        )
    except _PhysioRejection as e:
        # System-level rejection: simulation had < 80% valid patients.
        prefix = f"Sample {sample_order}: " if sample_order is not None else ""
        # print(
        #     f"{prefix}System rejected: only {e.valid_fraction:.3f} of trajectories "
        #     "fall within physiological normal ranges."
        # )
        return None

    # Compute summary statistics for the simulated trajectories.
    s_sim = compute_summary_stats(
        trajectories=trajectories,
        t_eval=t_eval,
        biomarker_names=biomarker_names,
        std_params=std_params,
    )
    mu_hat = s_sim

    residual = s_obs - mu_hat
    score = -float(np.sum(residual**2))

    if not np.isfinite(score):
        return None

    if verbose:
        print(f"\nQuadratic score ( -||s_obs - mu_hat||^2 ): {score:.2f}\n")

    return score
