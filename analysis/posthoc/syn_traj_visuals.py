"""Generate observed-vs-synthetic trajectory figures for a discovered ODE system.

Usage:
    python analysis/posthoc/syn_traj_visuals.py \
        --problem_name pkpd \
        --sample_order 70 \
        --log_path logs/pkpd_run1

Or, using observed initial conditions (one IC per patient, each replicated 5x):
    python analysis/posthoc/syn_traj_visuals.py \
        --problem_name pkpd \
        --sample_order 70 \
        --log_path logs/pkpd_run1 \
        --obs_ic

Figures are saved to: <LOG_PATH>/figures/sample<SAMPLE_ORDER>/
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import inspect
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentode.ode import initial_condition_utils  # noqa: E402
from agentode.ode import ode_simulator  # noqa: E402
from agentode.core import param_utils  # noqa: E402
from agentode.agent.figures import generate_observed_vs_synthetic_figures  # noqa: E402
from agentode.metrics.mnsd_score import evaluate_system_mnsd  # noqa: E402


def load_function_from_log(
    log_path: str,
    sample_order: int,
) -> tuple[Callable, Optional[dict]]:
    """Load an ODE `system` function and its parameter priors from the log."""
    json_path = os.path.join(log_path, "samples", f"samples_{sample_order}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Sample {sample_order} not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    function_str = data["function"]
    lowered = function_str.lower()
    is_torch_system = ("import torch" in lowered) or ("torch." in lowered)

    if is_torch_system:
        fixed_lines = []
        for line in function_str.split("\n"):
            if line.lstrip().startswith("#"):
                fixed_lines.append(line)
            else:
                line = re.sub(r'\bparams\[(\d+)\]', r'params[..., \1]', line)
                line = re.sub(r'\bbiomarkers\[(\d+)\]', r'biomarkers[..., \1]', line)
                fixed_lines.append(line)
        function_str = "\n".join(fixed_lines)

    param_distributions = data.get("param_distributions")

    namespace: dict = {"np": np}
    if is_torch_system:
        namespace["torch"] = torch
    exec(function_str, namespace)
    system_func = namespace.get("system")

    if callable(system_func):
        setattr(system_func, "_torch_backend", bool(is_torch_system))

    try:
        tree = ast.parse(function_str)
        n_states = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Name) and node.value.id == "biomarkers":
                    target = node.targets[0]
                    if isinstance(target, ast.Tuple):
                        n_states = len(target.elts)
                        break
        if n_states is not None and callable(system_func):
            setattr(system_func, "_state_dim", int(n_states))
    except SyntaxError:
        pass

    if not callable(system_func):
        raise ValueError(
            "Loaded function is not callable. Expected a definition like "
            "`def system(biomarkers, params[, t]): ...` in the log."
        )

    return system_func, param_distributions


def simulate_trajectories(
    system_func: Callable,
    problem_name: str,
    n_patients: Optional[int] = None,
    param_distributions: Optional[dict] = None,
    ic_array: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Simulate patient trajectories using the discovered ODE system."""
    config = initial_condition_utils.load_ic_config(problem_name)
    all_biomarker_names = initial_condition_utils.get_biomarker_order(config)
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    if ic_array is None:
        if n_patients is None:
            n_patients = config["simulation_params"]["default_sample_size"]
        ic_array, _ = initial_condition_utils.generate_initial_conditions(
            config,
            sample_size=n_patients,
            random_seed=config.get("random_seed", None),
        )
    else:
        n_patients = int(ic_array.shape[0])

    if param_distributions:
        params = param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n_patients,
            distribution="lognormal",
        )
    else:
        params = np.ones((n_patients, 1))

    use_torch_backend = bool(getattr(system_func, "_torch_backend", False))

    if use_torch_backend:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        traj_t = ode_simulator.simulate_ode_system_torch(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=t_eval,
            params=params,
            device=device,
            dtype=torch.float32,
            method="rk4",
        )
        trajectories = traj_t.detach().cpu().numpy()
    else:
        trajectories = ode_simulator.simulate_ode_system(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=t_eval,
            params=params,
            method="odeint",
        )

    valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
        trajectories,
        config=config,
        check_nans=True,
    )
    valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
    print(f"Fraction of physiologically valid trajectories: {valid_fraction:.3f}")

    trajectories = trajectories[valid_mask]
    observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]
    trajectories_obs = trajectories[..., observed_indices]

    records = []
    n_valid = trajectories_obs.shape[0]
    for patient_id in range(n_valid):
        for t_idx, t in enumerate(t_eval):
            record = {"id": patient_id, "t": t}
            for bio_idx, bio_name in enumerate(biomarker_names):
                record[bio_name] = trajectories_obs[patient_id, t_idx, bio_idx]
            records.append(record)

    df = pd.DataFrame(records)
    return df, biomarker_names, t_eval, trajectories_obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory figures from a discovered ODE system log."
    )
    parser.add_argument("--problem_name", required=True)
    parser.add_argument("--sample_order", type=int, required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--bin_width", type=float, default=None)
    parser.add_argument(
        "--obs_ic",
        action="store_true",
        help="Use observed baseline ICs (each replicated 5x) instead of config distributions.",
    )
    return parser.parse_args()


def _resolve_observed_path(problem_name: str) -> str:
    test_path = os.path.join("data", problem_name, f"{problem_name}_test.csv")
    default_path = os.path.join("data", problem_name, f"{problem_name}.csv")
    if os.path.exists(test_path):
        return test_path
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError(
        f"Observed data file not found. Looked for {test_path} and {default_path}."
    )


def main() -> None:
    args = parse_args()

    obs_path = _resolve_observed_path(args.problem_name)
    obs_df = pd.read_csv(obs_path)
    obs_df.columns = obs_df.columns.str.strip().str.lower()

    if "id" not in obs_df.columns or "t" not in obs_df.columns:
        raise ValueError(f"Expected columns 'id' and 't' in observed data at {obs_path}.")

    n_obs_patients = int(obs_df["id"].nunique())
    n_patients = 5 * n_obs_patients
    print(f"Observed patients: {n_obs_patients} | Simulating: {n_patients}")

    ic_array: Optional[np.ndarray] = None
    if args.obs_ic:
        config = initial_condition_utils.load_ic_config(args.problem_name)
        biomarker_order = initial_condition_utils.get_biomarker_order(config)
        baseline = obs_df.sort_values("t").groupby("id").first()
        biomarker_order_lower = [b.lower() for b in biomarker_order]
        missing = [b for b in biomarker_order if b.lower() not in baseline.columns]
        if missing:
            raise ValueError(f"Observed data missing biomarker columns: {missing}")
        baseline_ic = baseline[biomarker_order_lower].to_numpy(dtype=float)
        ic_array = np.repeat(baseline_ic, repeats=5, axis=0)

    print(f"Loading function from sample {args.sample_order}...")
    system_func, param_distributions = load_function_from_log(args.log_path, args.sample_order)

    inferred_n_states = getattr(system_func, "_state_dim", None)
    config_n_states = None
    try:
        ic_config = initial_condition_utils.load_ic_config(args.problem_name)
        config_n_states = len(initial_condition_utils.get_biomarker_order(ic_config))
    except FileNotFoundError:
        ic_config = None

    wrapped_system_func = system_func
    if inferred_n_states is not None and config_n_states is not None:
        if config_n_states > inferred_n_states:
            base_accepts_time = False
            try:
                sig = inspect.signature(system_func)
                param_names = list(sig.parameters.keys())
                if len(param_names) >= 3 and param_names[2] in ("t", "time", "time_point"):
                    base_accepts_time = True
            except (TypeError, ValueError):
                pass

            def padded_system(
                biomarkers: np.ndarray,
                params: np.ndarray,
                t: float | None = None,
                _base_func: Callable = system_func,
                _accepts_time: bool = base_accepts_time,
                _n_states: int = inferred_n_states,
            ) -> np.ndarray:
                core = biomarkers[:_n_states]
                core_deriv = _base_func(core, params, t) if _accepts_time else _base_func(core, params)
                core_deriv = np.asarray(core_deriv, dtype=float).reshape(-1)[:_n_states]
                deriv = np.zeros_like(biomarkers, dtype=float)
                deriv[:_n_states] = core_deriv
                return deriv

            print(
                f"State/config mismatch: system has {inferred_n_states} states, "
                f"IC config has {config_n_states}. Using padded wrapper."
            )
            wrapped_system_func = padded_system

    print(f"Simulating {n_patients} trajectories...")
    df, biomarker_names, t_eval, trajectories = simulate_trajectories(
        system_func=wrapped_system_func,
        problem_name=args.problem_name,
        n_patients=n_patients,
        param_distributions=param_distributions,
        ic_array=ic_array,
    )
    print(f"Biomarkers: {biomarker_names}")
    print(f"Time range: {t_eval[0]:.1f} - {t_eval[-1]:.1f}")

    try:
        mnsd = evaluate_system_mnsd(
            system_func=wrapped_system_func,
            problem_name=args.problem_name,
            param_distributions=param_distributions,
            verbose=False,
            sample_order=args.sample_order,
            backend="gpu" if bool(getattr(system_func, "_torch_backend", False)) else "cpu",
            standardization=True,
        )
        if mnsd is not None:
            print(f"MNSD: {mnsd:.4f}")
    except Exception as e:
        print(f"[syn_traj_visuals] Failed to compute MNSD: {e}")

    out_dir = os.path.join(args.log_path, "figures", f"sample{args.sample_order}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving figures to: {os.path.abspath(out_dir)}")

    fig_dict = generate_observed_vs_synthetic_figures(
        problem_name=args.problem_name,
        synthetic_df=df,
        bin_width=args.bin_width,
    )
    for key, png_bytes in fig_dict.items():
        out_path = os.path.join(out_dir, f"{key}.png")
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
