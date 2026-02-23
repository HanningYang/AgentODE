"""Euclidean summary-statistics distance for ODE systems.

Uses the same summary statistics as the synthetic likelihood module, but
replaces the log synthetic likelihood with a simple Euclidean distance:

    distance = ||s_obs - mu_hat||_2

where s_obs are summary statistics from observed data and mu_hat is the
mean summary-statistics vector from simulated trajectories.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from llmode.summary_stats import (
    compute_summary_stats,
    get_observed_summary,
)


N_PATIENTS_EUCLIDEAN = 1000


class _PhysioRejection(Exception):
    """Internal signal that a simulation failed physiological validity."""

    def __init__(self, valid_fraction: float):
        super().__init__(f"valid_fraction={valid_fraction:.3f}")
        self.valid_fraction = float(valid_fraction)


def evaluate_system_euclidean_distance(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    standardization: bool = False,
) -> float | None:
    """Evaluate a system via Euclidean distance in summary-stat space.

    By default, the quantile-based statistics are computed on *non-standardized*
    biomarker values (standardization=False), so this measures distance directly
    in the raw summary-statistics space.
    """
    from llmode import initial_condition_utils
    from llmode import ode_simulator
    from llmode import param_utils

    # If no parameter distributions are available, treat as unevaluable.
    if param_distributions is None:
        return None

    # Load configuration and observed data (cached per problem and
    # standardization choice).
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
            # Impose a 80% plausibility check for simulated data
            if valid_fraction < 0.8:
                if sample_order not in (0, None):
                    # Signal to the outer evaluator that this system should be rejected.
                    raise _PhysioRejection(valid_fraction)

            # Map observed biomarker names 
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
        # Single simulation with N_PATIENTS_EUCLIDEAN patients.
        trajectories = simulator(
            N_PATIENTS_EUCLIDEAN,
            param_sampler,
            t_eval,
        )
    except _PhysioRejection:
        # System-level rejection: simulation had < 80% valid patients.
        return None

    # Compute summary statistics for the simulated trajectories.
    s_sim = compute_summary_stats(
        trajectories=trajectories,
        t_eval=t_eval,
        biomarker_names=biomarker_names,
        std_params=std_params,
        standardization=standardization,
    )
    mu_hat = s_sim

    residual = s_obs - mu_hat
    distance = float(np.linalg.norm(residual))

    if not np.isfinite(distance):
        return None

    if verbose:
        print(f"\nEuclidean distance (||s_obs - mu_hat||_2): {distance:.2f}\n")

    return distance

