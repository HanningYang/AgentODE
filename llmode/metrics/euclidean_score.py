"""Euclidean summary-statistics distance for ODE systems.

Computes distance = ||s_obs - mu_hat||_2 where s_obs are observed summary
statistics and mu_hat is the mean summary-statistics vector from simulations.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from llmode.metrics.summary_stats import (
    compute_summary_stats,
    get_observed_summary,
)


N_PATIENTS_EUCLIDEAN = 1000


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
    std_params = obs["std_params"]
    s_obs = obs["s_obs"]

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

    s_sim = compute_summary_stats(
        trajectories=trajectories,
        t_eval=t_eval,
        biomarker_names=biomarker_names,
        std_params=std_params,
        standardization=standardization,
    )

    distance = float(np.linalg.norm(s_obs - s_sim))

    if not np.isfinite(distance):
        return None

    if verbose:
        print(f"Euclidean distance (||s_obs - mu_hat||_2): {distance:.2f}")

    return distance
