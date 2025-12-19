"""Utilities for simulating ODE systems with generated initial conditions."""

import os

import numpy as np
from scipy.integrate import odeint
from typing import Callable, Dict, Optional, Tuple
from llmode import initial_condition_utils
from llmode import code_manipulation


_NUMERICAL_WARNING_PRINTED = False


def _silence_odeint_output():
    """Context manager to silence low-level LSODA prints.

    LSODA (Fortran code behind `odeint`) writes directly to C-level stdout/stderr,
    which is not affected by ``contextlib.redirect_stdout/stderr``. This context
    temporarily redirects file descriptors 1 and 2 to ``os.devnull`` so that
    LSODA warnings are not printed.
    """
    class _Silencer:
        def __enter__(self):
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            self._stdout_fd = os.dup(1)
            self._stderr_fd = os.dup(2)
            os.dup2(self._devnull_fd, 1)
            os.dup2(self._devnull_fd, 2)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            os.dup2(self._stdout_fd, 1)
            os.dup2(self._stderr_fd, 2)
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)
            os.close(self._devnull_fd)

    return _Silencer()


def simulate_ode_system(
    system_func: Callable,
    initial_conditions: np.ndarray,
    time_grid: np.ndarray,
    params: np.ndarray,
    method: str = 'odeint'
) -> np.ndarray:
    """Simulate an ODE system for multiple initial conditions.

    Args:
        system_func: Function with signature (biomarkers, params) -> derivatives
        initial_conditions: Array of shape (n_samples, n_biomarkers) with initial values
        time_grid: Array of time points for simulation
        params: Array of parameter values for the ODE system. Can be:
            - 1D array of shape (n_params,) used for all samples
            - 2D array of shape (n_samples, n_params) providing a distinct
              parameter set for each sample
        method: Integration method ('odeint' for scipy.integrate.odeint)

    Returns:
        Array of shape (n_samples, n_timepoints, n_biomarkers) with trajectories

    Example:
        >>> def system(biomarkers, params):
        ...     return np.array([params[0] - params[1] * biomarkers[0]])
        >>> ic = np.array([[1.0], [2.0]])  # 2 samples, 1 biomarker
        >>> t = np.linspace(0, 10, 100)
        >>> params = np.array([0.5, 0.1])
        >>> trajectories = simulate_ode_system(system, ic, t, params)
        >>> print(trajectories.shape)  # (2, 100, 1)
    """
    n_samples = initial_conditions.shape[0]
    n_timepoints = len(time_grid)
    n_biomarkers = initial_conditions.shape[1]

    trajectories = np.zeros((n_samples, n_timepoints, n_biomarkers))

    # Normalise parameter shape: allow either shared params or per-sample params.
    params = np.asarray(params)
    if params.ndim == 1:
        # Broadcast a single parameter vector to all samples.
        params_per_sample = np.repeat(params[None, :], n_samples, axis=0)
    elif params.ndim == 2:
        if params.shape[0] != n_samples:
            raise ValueError(
                f"params has shape {params.shape}, but n_samples={n_samples}. "
                "When passing per-patient parameter sets, params should be "
                "(n_samples, n_params)."
            )
        params_per_sample = params
    else:
        raise ValueError(
            f"Unsupported params shape {params.shape}; expected 1D or 2D array."
        )

    if method == 'odeint':
        global _NUMERICAL_WARNING_PRINTED
        for i in range(n_samples):
            # Wrapper to match odeint's expected signature: f(y, t)
            def ode_func(y, t, local_params=params_per_sample[i]):
                return system_func(y, local_params)

            try:
                # Completely silence LSODA's internal warning prints for this call.
                with _silence_odeint_output():
                    trajectory = odeint(ode_func, initial_conditions[i], time_grid)
                trajectories[i] = trajectory

                # If trajectories look numerically extreme or invalid, emit
                # a single high-level warning so the user is aware, without
                # flooding the console with LSODA's internal messages.
                if not _NUMERICAL_WARNING_PRINTED:
                    if np.any(np.isnan(trajectory)) or np.nanmax(np.abs(trajectory)) > 1e3:
                        print(
                            "Warning: ODE integration produced extreme or invalid "
                            "values; results may be numerically unreliable."
                        )
                        _NUMERICAL_WARNING_PRINTED = True
            except Exception as e:
                # print(f"Warning: Integration failed for sample {i}: {e}")
                trajectories[i] = np.nan

    else:
        raise ValueError(f"Unsupported integration method: {method}")

    return trajectories


def simulate_ode_system_with_resampling(
    system_func: Callable,
    initial_conditions: np.ndarray,
    time_grid: np.ndarray,
    param_sampler: Callable[[int], np.ndarray],
    config: dict,
    max_param_draws_per_sample: int = 5,
    method: str = 'odeint',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an ODE system ensuring non-negative trajectories via per-sample resampling.

    Each simulated patient is assigned its own parameter set drawn from ``param_sampler``.
    If a trajectory contains negative values or violates bounds, that parameter set is
    discarded and a new one is drawn for the same patient (up to ``max_param_draws_per_sample``).

    If no valid parameter set can be found for a patient, its trajectory is filled with NaNs
    and the corresponding row in the returned ``valid_mask`` is False. Callers can treat
    the case ``valid_mask.any() is False`` as "ODE system not suitable" and skip evaluation
    (e.g. return ``None`` as the score).

    Args:
        system_func: Function with signature (biomarkers, params) -> derivatives.
        initial_conditions: Array of shape (n_samples, n_biomarkers).
        time_grid: Array of time points for simulation.
        param_sampler: Callable taking the sample index and returning a parameter vector.
        config: Initial-condition configuration dictionary (used for bounds and biomarkers).
        max_param_draws_per_sample: Maximum number of parameter draws per patient.
        method: Integration method ('odeint').

    Returns:
        Tuple of:
            - trajectories: Array of shape (n_samples, n_timepoints, n_biomarkers).
            - params_used: Array of shape (n_samples, n_params) with the accepted
              parameter set for each sample (NaNs where no valid set was found).
            - valid_mask: Boolean array of shape (n_samples,) indicating which
              trajectories are valid.
    """
    n_samples = initial_conditions.shape[0]
    n_timepoints = len(time_grid)
    n_biomarkers = initial_conditions.shape[1]

    trajectories = np.full((n_samples, n_timepoints, n_biomarkers), np.nan)
    params_used_list = []
    valid_mask = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        accepted = False
        params_i = None
        # Track reasons for rejection across draws for this sample.
        rejection_reasons = {
            'has_nan': 0,
            'negative_values': 0,
            'exceeds_bounds': 0,
        }

        for draw in range(max_param_draws_per_sample):
            params_candidate = np.asarray(param_sampler(i))

            # Simulate a single-patient trajectory with this parameter set.
            traj_i = simulate_ode_system(
                system_func=system_func,
                initial_conditions=initial_conditions[i : i + 1],
                time_grid=time_grid,
                params=params_candidate,
                method=method,
            )

            # traj_i has shape (1, n_timepoints, n_biomarkers)
            single_valid_mask, _issues = check_trajectory_validity(
                traj_i,
                config=config,
                check_bounds=False,  # only enforce NaNs/negativity during resampling
                check_nans=True,
            )

            if bool(single_valid_mask[0]):
                trajectories[i] = traj_i[0]
                params_i = params_candidate
                valid_mask[i] = True
                accepted = True
                break
            else:
                # Inspect the trajectory directly to understand why it failed.
                traj_single = traj_i[0]
                if np.any(np.isnan(traj_single)):
                    rejection_reasons['has_nan'] += 1
                elif np.any(traj_single < 0):
                    rejection_reasons['negative_values'] += 1
                else:
                    # Check bounds per biomarker.
                    biomarker_names = initial_condition_utils.get_biomarker_order(config)
                    for j, name in enumerate(biomarker_names):
                        biomarker_config = config['biomarkers'][name]
                        clip_min = biomarker_config['clip_min']
                        clip_max = biomarker_config['clip_max']
                        if (np.any(traj_single[:, j] < clip_min) or
                                np.any(traj_single[:, j] > clip_max)):
                            rejection_reasons['exceeds_bounds'] += 1
                            break

        # if not accepted:
        #     print(
        #         f"Warning: No valid parameter set found for sample {i} "
        #         f"after {max_param_draws_per_sample} draws."
        #     )
        #     print(f"  Rejection summary for sample {i}: "
        #           f"NaN: {rejection_reasons['has_nan']}, "
        #           f"negative: {rejection_reasons['negative_values']}, "
        #           f"out_of_bounds: {rejection_reasons['exceeds_bounds']}")
            # params_i stays None and will be converted to NaNs below.
        params_used_list.append(params_i)

    # Convert params_used_list (possibly containing None) into a 2D array with NaNs.
    if any(p is not None for p in params_used_list):
        # Determine parameter dimension from first non-None entry.
        first = next(p for p in params_used_list if p is not None)
        n_params = first.shape[0]
        params_used = np.full((n_samples, n_params), np.nan)
        for i, p in enumerate(params_used_list):
            if p is not None:
                params_used[i] = p
    else:
        params_used = np.empty((n_samples, 0))

    if verbose:
        n_valid = int(valid_mask.sum())
        print("\n[Simulation Summary]")
        print(f"  Patients (samples)        : {n_samples}")
        print(f"  Time points per trajectory: {n_timepoints}")
        print(f"  Biomarkers per patient    : {n_biomarkers}")
        print(f"  Valid trajectories        : {n_valid} / {n_samples}")
        if n_valid > 0:
            global_min = float(np.nanmin(trajectories))
            global_max = float(np.nanmax(trajectories))
            print(f"  Trajectory value range    : [{global_min:.4g}, {global_max:.4g}]")
        print("============================================\n")

    return trajectories, params_used, valid_mask


def simulate_from_config(
    system_func: Callable,
    params: np.ndarray,
    config: dict,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Simulate ODE system using configuration file for initial conditions.

    Args:
        system_func: ODE system function with signature (biomarkers, params) -> derivatives
        params: Parameter values for the ODE system
        config: Configuration dictionary from initial_condition_utils.load_ic_config()
        sample_size: Number of samples to simulate (uses config default if None)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of:
            - trajectories: Array of shape (n_samples, n_timepoints, n_biomarkers)
            - time_grid: Array of time points
            - initial_conditions_dict: Dictionary mapping biomarker names to initial values

    Example:
        >>> config = initial_condition_utils.load_ic_config('aki')
        >>> def system(biomarkers, params):
        ...     return params[0] - params[1] * biomarkers
        >>> params = np.array([1.0, 0.1])
        >>> trajectories, t, ic_dict = simulate_from_config(system, params, config, sample_size=100)
    """
    # Generate initial conditions
    ic_array, ic_dict = initial_condition_utils.generate_initial_conditions(
        config,
        sample_size=sample_size,
        random_seed=random_seed
    )

    if verbose:
        print("\n[Initial Conditions]")
        print(f"  Patients (samples) : {ic_array.shape[0]}")
        print(f"  Biomarkers         : {ic_array.shape[1]}")

    # Get time grid
    time_grid = initial_condition_utils.get_time_grid(config)

    if verbose:
        print("\n[Time Grid]")
        print(f"  Points: {len(time_grid)}")
        print(f"  Range : [{time_grid[0]}, {time_grid[-1]}]")

    # Simulate
    trajectories = simulate_ode_system(system_func, ic_array, time_grid, params)

    if verbose:
        n_samples, n_timepoints, n_biomarkers = trajectories.shape
        print("\n[Trajectories]")
        print(f"  Shape              : {trajectories.shape}")
        print(f"  Patients           : {n_samples}")
        print(f"  Time points        : {n_timepoints}")
        print(f"  Biomarkers         : {n_biomarkers}")
        print(f"  NaN count          : {int(np.isnan(trajectories).sum())}")
        print(f"  Value range (finite): "
              f"[{float(np.nanmin(trajectories)):.4g}, "
              f"{float(np.nanmax(trajectories)):.4g}]")
        print("============================================\n")

    return trajectories, time_grid, ic_dict


def simulate_from_function_object(
    function_obj: code_manipulation.Function,
    params: np.ndarray,
    config: dict,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Simulate ODE system from a Function object (e.g., from LLM generation).

    Args:
        function_obj: Function object with executable code
        params: Parameter values for the ODE system
        config: Configuration dictionary
        sample_size: Number of samples to simulate
        random_seed: Random seed

    Returns:
        Tuple of (trajectories, time_grid, initial_conditions_dict)

    Example:
        >>> # After LLM generates a function
        >>> from llmode import code_manipulation
        >>> program = code_manipulation.text_to_program(llm_generated_code)
        >>> func = program.get_function('system')
        >>> config = initial_condition_utils.load_ic_config('aki')
        >>> params = np.array([1.0, 0.1, ...])
        >>> trajectories, t, ic = simulate_from_function_object(func, params, config)
    """
    # Get the executable function
    system_func = function_obj.to_callable()

    return simulate_from_config(
        system_func=system_func,
        params=params,
        config=config,
        sample_size=sample_size,
        random_seed=random_seed,
        verbose=verbose,
    )


def compute_trajectory_statistics(
    trajectories: np.ndarray,
    axis: int = 0
) -> Dict[str, np.ndarray]:
    """Compute statistics across trajectory samples.

    Args:
        trajectories: Array of shape (n_samples, n_timepoints, n_biomarkers)
        axis: Axis to compute statistics over (0 for across samples)

    Returns:
        Dictionary with keys 'mean', 'std', 'median', 'q25', 'q75'
        Each value is array of shape (n_timepoints, n_biomarkers)
    """
    stats = {
        'mean': np.nanmean(trajectories, axis=axis),
        'std': np.nanstd(trajectories, axis=axis),
        'median': np.nanmedian(trajectories, axis=axis),
        'q25': np.nanpercentile(trajectories, 25, axis=axis),
        'q75': np.nanpercentile(trajectories, 75, axis=axis),
        'min': np.nanmin(trajectories, axis=axis),
        'max': np.nanmax(trajectories, axis=axis)
    }

    return stats


def check_trajectory_validity(
    trajectories: np.ndarray,
    config: dict,
    check_bounds: bool = True,
    check_nans: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Check validity of simulated trajectories.

    Args:
        trajectories: Array of shape (n_samples, n_timepoints, n_biomarkers)
        config: Configuration dictionary with biomarker definitions
        check_bounds: Whether to check if values stay within physiological ranges
        check_nans: Whether to check for NaN values

    Returns:
        Tuple of:
            - valid_mask: Boolean array of shape (n_samples,) indicating valid trajectories
            - issue_counts: Dictionary with counts of different issues

    Example:
        >>> valid_mask, issues = check_trajectory_validity(trajectories, config)
        >>> print(f"Valid trajectories: {valid_mask.sum()} / {len(valid_mask)}")
        >>> print(f"Issues: {issues}")
    """
    n_samples = trajectories.shape[0]
    valid_mask = np.ones(n_samples, dtype=bool)
    issue_counts = {
        'has_nan': 0,
        'exceeds_bounds': 0,
        'negative_values': 0
    }

    biomarker_names = initial_condition_utils.get_biomarker_order(config)

    for i in range(n_samples):
        trajectory = trajectories[i]

        # Check for NaNs
        if check_nans and np.any(np.isnan(trajectory)):
            valid_mask[i] = False
            issue_counts['has_nan'] += 1
            continue

        # Check for negative values (biomarkers should be positive)
        if np.any(trajectory < 0):
            valid_mask[i] = False
            issue_counts['negative_values'] += 1
            continue

        # Check bounds
        if check_bounds:
            for j, name in enumerate(biomarker_names):
                biomarker_config = config['biomarkers'][name]
                clip_min = biomarker_config['clip_min']
                clip_max = biomarker_config['clip_max']

                if np.any(trajectory[:, j] < clip_min) or np.any(trajectory[:, j] > clip_max):
                    valid_mask[i] = False
                    issue_counts['exceeds_bounds'] += 1
                    break

    return valid_mask, issue_counts


def check_trajectory_normal_range_validity(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Check validity using physiological normal ranges for each biomarker.

    A trajectory is considered invalid if:
      - It contains any NaN values, or
      - It contains any negative values, or
      - Any biomarker leaves its physiological normal range
        [normal_min, normal_max] (if defined).
    """
    n_samples = trajectories.shape[0]
    valid_mask = np.ones(n_samples, dtype=bool)
    issue_counts = {
        'has_nan': 0,
        'negative_values': 0,
        'outside_normal_range': 0,
    }

    biomarker_names = initial_condition_utils.get_biomarker_order(config)

    for i in range(n_samples):
        trajectory = trajectories[i]

        # Check for NaNs
        if check_nans and np.any(np.isnan(trajectory)):
            valid_mask[i] = False
            issue_counts['has_nan'] += 1
            continue

        # Check for negative values (biomarkers should be non-negative)
        if np.any(trajectory < 0):
            valid_mask[i] = False
            issue_counts['negative_values'] += 1
            continue

        # Check physiological normal ranges when available.
        for j, name in enumerate(biomarker_names):
            phys_range = config['biomarkers'][name].get('physiological_range')
            if not phys_range:
                continue
            normal_min = phys_range.get('normal_min')
            normal_max = phys_range.get('normal_max')
            if normal_min is None or normal_max is None:
                continue

            vals = trajectory[:, j]
            if np.any(vals < normal_min) or np.any(vals > normal_max):
                valid_mask[i] = False
                issue_counts['outside_normal_range'] += 1
                break

    return valid_mask, issue_counts


def save_trajectories(
    trajectories: np.ndarray,
    time_grid: np.ndarray,
    initial_conditions: Dict[str, np.ndarray],
    params: np.ndarray,
    config: dict,
    output_path: str
):
    """Save simulated trajectories to a compressed numpy file.

    Args:
        trajectories: Array of shape (n_samples, n_timepoints, n_biomarkers)
        time_grid: Array of time points
        initial_conditions: Dictionary of initial condition arrays
        params: Parameter values used
        config: Configuration dictionary
        output_path: Path to save file (should end in .npz)

    Example:
        >>> save_trajectories(traj, t, ic, params, config, 'output/aki_trajectories.npz')
        >>> # Load later with: data = np.load('output/aki_trajectories.npz')
    """
    np.savez_compressed(
        output_path,
        trajectories=trajectories,
        time_grid=time_grid,
        params=params,
        problem_name=config['problem_name'],
        biomarker_names=initial_condition_utils.get_biomarker_order(config),
        **initial_conditions  # Save each biomarker's IC as a separate array
    )

    print(f"Saved trajectories to {output_path}")
    print(f"  Shape: {trajectories.shape}")
    print(f"  Size: {trajectories.nbytes / 1024 / 1024:.2f} MB")
