"""Utilities for simulating ODE systems with generated initial conditions."""

import os
import inspect

import numpy as np
from scipy.integrate import odeint
from typing import Callable, Dict, Optional, Tuple
from agentode.ode import initial_condition_utils
from agentode.core import code_manipulation
import torch


_NUMERICAL_WARNING_PRINTED = False
_TORCH_DEVICE_LOGGED = False


def _silence_odeint_output():
    """Redirect fd 1/2 to devnull to suppress LSODA's C-level stdout/stderr."""

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


def check_trajectory_normal_range_validity(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Check validity using physiological normal ranges for each biomarker.

    A trajectory is marked invalid if it has any NaN, inf, negative value,
    or any biomarker outside its [normal_min, normal_max] physiological
    range (if defined). All issue types present in a trajectory are counted;
    i.e. a single trajectory can contribute to multiple counters.
    """
    n_samples = trajectories.shape[0]
    valid_mask = np.ones(n_samples, dtype=bool)
    issue_counts: Dict[str, int] = {
        "has_nan": 0,
        "has_inf": 0,
        "negative_values": 0,
        "outside_normal_range": 0,
    }

    biomarker_names = initial_condition_utils.get_biomarker_order(config)
    # Optional global flag: if config["negative"] is True, allow negatives
    # by default unless overridden at the biomarker level.
    global_allow_negative = bool(config.get("negative", False))

    for i in range(n_samples):
        trajectory = trajectories[i]

        sample_invalid = False

        if check_nans and np.any(np.isnan(trajectory)):
            sample_invalid = True
            issue_counts["has_nan"] += 1

        if np.any(np.isinf(trajectory)):
            sample_invalid = True
            issue_counts["has_inf"] += 1

        # Check for disallowed negative values per biomarker.
        has_disallowed_negative = False
        for j, name in enumerate(biomarker_names):
            allow_negative = config["biomarkers"][name].get(
                "negative", global_allow_negative
            )
            if allow_negative:
                continue
            vals = trajectory[:, j]
            if np.any(vals < 0):
                has_disallowed_negative = True
                break

        if has_disallowed_negative:
            sample_invalid = True
            issue_counts["negative_values"] += 1

        outside_range = False
        for j, name in enumerate(biomarker_names):
            phys_range = config["biomarkers"][name].get("physiological_range")
            if not phys_range:
                continue
            normal_min = phys_range.get("normal_min")
            normal_max = phys_range.get("normal_max")
            if normal_min is None or normal_max is None:
                continue
            vals = trajectory[:, j]
            if np.any(vals < normal_min) or np.any(vals > normal_max):
                outside_range = True
                break

        if outside_range:
            sample_invalid = True
            issue_counts["outside_normal_range"] += 1

        if sample_invalid:
            valid_mask[i] = False

    return valid_mask, issue_counts


def _to_tensor(x, device: torch.device, dtype: torch.dtype):
    """Convert NumPy or torch tensors to torch.Tensor on the given device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def simulate_ode_system_torch(
    system_func: Callable[..., torch.Tensor],
    initial_conditions,
    time_grid,
    params,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    method: str = "rk4",
):
    """Simulate an ODE system in torch for multiple initial conditions (RK4).

    Args:
        system_func: (state, params[, t]) -> derivatives, state shape (batch, n_biomarkers).
        initial_conditions: (n_samples, n_biomarkers).
        time_grid: 1D array of time points.
        params: (n_params,) or (n_samples, n_params).
        device: torch.device (defaults to CUDA if available).
        dtype: Torch dtype for computation.
        method: Only 'rk4' is implemented.

    Returns:
        Tensor of shape (n_samples, n_timepoints, n_biomarkers).
    """
    global _TORCH_DEVICE_LOGGED

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _TORCH_DEVICE_LOGGED:
        print(f"[torch ODE] using device: {device}")
        _TORCH_DEVICE_LOGGED = True

    t_grid = _to_tensor(time_grid, device, dtype).flatten()
    y0 = _to_tensor(initial_conditions, device, dtype)
    params_tensor = _to_tensor(params, device, dtype)

    if params_tensor.ndim == 1:
        params_tensor = params_tensor.unsqueeze(0).expand(y0.shape[0], -1)
    elif params_tensor.ndim == 2 and params_tensor.shape[0] != y0.shape[0]:
        raise ValueError(
            f"params has shape {tuple(params_tensor.shape)}, but n_samples={y0.shape[0]}"
        )

    # Detect whether system_func expects an explicit time argument.
    accepts_time = False
    try:
        sig = inspect.signature(system_func)
        if len(sig.parameters) >= 3:
            accepts_time = True
    except (TypeError, ValueError):
        accepts_time = False

    n_steps = t_grid.shape[0]
    n_samples, n_biomarkers = y0.shape

    traj = torch.empty((n_samples, n_steps, n_biomarkers), device=device, dtype=dtype)
    traj[:, 0] = y0

    if method != "rk4":
        raise ValueError(f"Unsupported method '{method}', only 'rk4' is implemented.")

    y = y0
    for i in range(1, n_steps):
        t_prev = t_grid[i - 1]
        t_curr = t_grid[i]
        dt = (t_curr - t_prev).item()

        if accepts_time:
            k1 = system_func(y, params_tensor, t_prev)
            k2 = system_func(y + 0.5 * dt * k1, params_tensor, t_prev + 0.5 * dt)
            k3 = system_func(y + 0.5 * dt * k2, params_tensor, t_prev + 0.5 * dt)
            k4 = system_func(y + dt * k3, params_tensor, t_curr)
        else:
            k1 = system_func(y, params_tensor)
            k2 = system_func(y + 0.5 * dt * k1, params_tensor)
            k3 = system_func(y + 0.5 * dt * k2, params_tensor)
            k4 = system_func(y + dt * k3, params_tensor)

        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[:, i] = y

    return traj


def simulate_ode_system(
    system_func: Callable,
    initial_conditions: np.ndarray,
    time_grid: np.ndarray,
    params: np.ndarray,
    method: str = 'odeint'
) -> np.ndarray:
    """Simulate an ODE system for multiple initial conditions.

    Args:
        system_func: (biomarkers, params[, t]) -> derivatives.
        initial_conditions: (n_samples, n_biomarkers).
        time_grid: Array of time points.
        params: (n_params,) shared across samples, or (n_samples, n_params) per-sample.
        method: Integration method ('odeint').

    Returns:
        Array of shape (n_samples, n_timepoints, n_biomarkers).
    """
    n_samples = initial_conditions.shape[0]
    n_timepoints = len(time_grid)
    n_biomarkers = initial_conditions.shape[1]

    trajectories = np.zeros((n_samples, n_timepoints, n_biomarkers))

    params = np.asarray(params)
    if params.ndim == 1:
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

    # Support both (biomarkers, params) and (biomarkers, params, t) signatures.
    accepts_time = False
    try:
        sig = inspect.signature(system_func)
        param_names = list(sig.parameters.keys())
        if len(param_names) >= 3 and param_names[2] in ('t', 'time', 'time_point'):
            accepts_time = True
    except (TypeError, ValueError):
        accepts_time = False

    if method == 'odeint':
        global _NUMERICAL_WARNING_PRINTED
        for i in range(n_samples):
            def ode_func(y, t, local_params=params_per_sample[i], use_time=accepts_time):
                if use_time:
                    return system_func(y, local_params, t)
                return system_func(y, local_params)

            try:
                with _silence_odeint_output():
                    trajectory = odeint(ode_func, initial_conditions[i], time_grid)
                trajectories[i] = trajectory

                if not _NUMERICAL_WARNING_PRINTED:
                    if np.any(np.isnan(trajectory)) or np.nanmax(np.abs(trajectory)) > 1e3:
                        print(
                            "Warning: ODE integration produced extreme or invalid "
                            "values; results may be numerically unreliable."
                        )
                        _NUMERICAL_WARNING_PRINTED = True
            except Exception:
                trajectories[i] = np.nan

    else:
        raise ValueError(f"Unsupported integration method: {method}")

    return trajectories


def simulate_from_config(
    system_func: Callable,
    params: np.ndarray,
    config: dict,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Simulate ODE system using a config file for initial conditions."""
    ic_array, ic_dict = initial_condition_utils.generate_initial_conditions(
        config,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    time_grid = initial_condition_utils.get_time_grid(config)
    # Respect torch vs NumPy backend in the same way as `simulate_valid_from_config`.
    use_torch_backend = bool(getattr(system_func, "_torch_backend", False))
    if use_torch_backend:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        traj_t = simulate_ode_system_torch(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=time_grid,
            params=params,
            device=device,
            dtype=torch.float32,
            method="rk4",
        )
        trajectories = traj_t.detach().cpu().numpy()
    else:
        trajectories = simulate_ode_system(
            system_func, ic_array, time_grid, params
        )
    return trajectories, time_grid, ic_dict


def simulate_valid_from_config(
    system_func: Callable,
    params: np.ndarray,
    config: dict,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    check_nans: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    """Simulate from config and drop numerically/physiologically invalid trajectories."""
    ic_array, ic_dict = initial_condition_utils.generate_initial_conditions(
        config,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    time_grid = initial_condition_utils.get_time_grid(config)

    use_torch_backend = bool(getattr(system_func, "_torch_backend", False))
    if use_torch_backend:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        traj_t = simulate_ode_system_torch(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=time_grid,
            params=params,
            device=device,
            dtype=torch.float32,
            method="rk4",
        )
        trajectories = traj_t.detach().cpu().numpy()
    else:
        trajectories = simulate_ode_system(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=time_grid,
            params=params,
            method="odeint",
        )

    valid_mask, issues = check_trajectory_normal_range_validity(
        trajectories=trajectories,
        config=config,
        check_nans=check_nans,
    )
    return trajectories[valid_mask], time_grid, ic_dict, valid_mask, issues
