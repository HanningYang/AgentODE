"""Visualize simulated ODE trajectories from discovered equations.

Usage:
    python analysis/posthoc/evaluate_and_visualize_system.py --problem_name aki --sample_order 42 --log_path logs/aki_run1
"""

import argparse
import ast
import json
import os
import sys
import inspect
from typing import Callable, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentode.core import code_manipulation
from agentode.ode import initial_condition_utils
from agentode.ode import ode_simulator
from agentode.core import param_utils
from agentode.metrics.summary_stats import (
    compute_standardization_params,
    compute_summary_stats_from_df,
    compute_summary_stats,
    get_stat_names,
)
from agentode.metrics.mntd_score import evaluate_system_mntd


def load_function_from_log(
    log_path: str,
    sample_order: int,
    verbose: bool = True,
) -> Tuple[Callable, Optional[dict]]:
    """Load an ODE `system` function and its parameter priors from the log.

    Returns:
        system_func: Callable with signature (biomarkers, params) -> derivatives
        param_distributions: Optional dict of parameter distributions stored in the log
    """
    json_path = os.path.join(log_path, 'samples', f'samples_{sample_order}.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Sample {sample_order} not found at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    function_str = data['function']
    lowered = function_str.lower()
    is_torch_system = ("import torch" in lowered) or ("torch." in lowered)

    if verbose:
        print('Function:')
        print(function_str)
        print(64*'-')
    param_distributions = data.get('param_distributions')

    # Execute the function definition in an isolated namespace.
    namespace: dict = {"np": np}
    if is_torch_system:
        namespace["torch"] = torch
    exec(function_str, namespace)
    system_func = namespace.get("system")

    if callable(system_func):
        setattr(system_func, "_torch_backend", bool(is_torch_system))

    # Infer number of state variables from `x1, x2, ... = biomarkers` unpacking.
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
    n_patients: int = 1000,
    param_distributions: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Simulate patient trajectories using the discovered ODE system.

    Returns:
        df: DataFrame with columns [id, t, biomarker1, biomarker2, ...]
        biomarker_names: List of biomarker column names
        t_eval: Time grid used for simulation
    """
    config = initial_condition_utils.load_ic_config(problem_name)
    all_biomarker_names = initial_condition_utils.get_biomarker_order(config)
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    ic_array, _ = initial_condition_utils.generate_initial_conditions(
        config,
        sample_size=n_patients,
        random_seed=config.get('random_seed', None),
    )

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
            method='odeint',
        )

    valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
        trajectories, config=config, check_nans=True,
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
            record = {'id': patient_id, 't': t}
            for bio_idx, bio_name in enumerate(biomarker_names):
                record[bio_name] = trajectories_obs[patient_id, t_idx, bio_idx]
            records.append(record)

    df = pd.DataFrame(records)
    return df, biomarker_names, t_eval, trajectories_obs


def create_visualizations(
    df: pd.DataFrame,
    biomarker_names: list[str],
    save_path: str = None,
    y_ranges: dict | None = None,
    bin_width: float | None = None,
):
    """Create comprehensive visualizations for simulated trajectories."""
    n_biomarkers = len(biomarker_names)
    fig = plt.figure(figsize=(18, 12))

    sample_patients = df['id'].unique()[:5]

    # Row 1: Individual patient trajectories
    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, i + 1)
        for patient_id in sample_patients:
            patient_data = df[df['id'] == patient_id]
            ax.plot(patient_data['t'], patient_data[biomarker],
                   marker='o', alpha=0.6, label=f'Patient {patient_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel(biomarker)
        ax.set_title(f'{biomarker} Trajectories (Sample)')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if y_ranges and 'trajectories' in y_ranges:
            ylim = y_ranges['trajectories'].get(biomarker)
            if ylim is not None:
                ax.set_ylim(*ylim)

    # Row 2: Aggregate temporal trends with confidence intervals
    t_min = float(df['t'].min())
    t_max = float(df['t'].max())
    if bin_width is None:
        bin_width = (t_max - t_min) / 10.0
    time_bins = np.arange(t_min, t_max + bin_width, bin_width)
    df['time_bin'] = pd.cut(df['t'], bins=time_bins)

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, n_biomarkers + i + 1)
        # observed=False avoids FutureWarning with categorical groupby
        stats = df.groupby('time_bin', observed=False)[biomarker].agg(['mean', 'std', 'count'])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        stats['t_critical'] = stats['count'].apply(lambda n: scipy.stats.t.ppf(0.975, n-1))
        stats['ci'] = stats['t_critical'] * stats['se']
        bin_centers = [(interval.left + interval.right) / 2 for interval in stats.index]

        ax.plot(bin_centers, stats['mean'], linewidth=2, label='Mean')
        ax.fill_between(bin_centers,
                        stats['mean'] - stats['ci'],
                        stats['mean'] + stats['ci'],
                        alpha=0.3, label='95% CI')
        ax.set_xlabel('Time')
        ax.set_ylabel(biomarker)
        ax.set_title(f'{biomarker}: Mean ± 95% CI')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        if y_ranges and 'mean_ci' in y_ranges:
            ylim = y_ranges['mean_ci'].get(biomarker)
            if ylim is not None:
                ax.set_ylim(*ylim)

    # Row 3: Distribution density heatmaps
    time_bins_fine = np.linspace(t_min, t_max, 30)
    colormaps = ['YlOrRd', 'YlGn', 'YlGnBu', 'PuRd', 'BuPu', 'OrRd']

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, 2 * n_biomarkers + i + 1)
        valid_data = df[['t', biomarker]].dropna()

        if len(valid_data) > 0:
            if y_ranges and 'density' in y_ranges and biomarker in y_ranges['density']:
                y_min, y_max = y_ranges['density'][biomarker]
            else:
                y_min = float(valid_data[biomarker].min())
                y_max = float(valid_data[biomarker].max())

            bio_bins = np.linspace(y_min, y_max, 30)
            H, xedges, yedges = np.histogram2d(
                valid_data['t'],
                valid_data[biomarker],
                bins=[time_bins_fine, bio_bins]
            )

            im = ax.imshow(H.T, origin='lower', aspect='auto',
                          cmap=colormaps[i % len(colormaps)],
                          extent=[t_min, t_max, y_min, y_max])
            ax.set_xlabel('Time')
            ax.set_ylabel(biomarker)
            ax.set_title(f'{biomarker} Density over Time')
            ax.set_ylim(y_min, y_max)
            plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize simulated trajectories from discovered ODE equations'
    )
    parser.add_argument('--problem_name', type=str, required=True)
    parser.add_argument('--sample_order', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--n_patients', type=int, default=1000)
    parser.add_argument('--param_dist_path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    print(f"Loading function from sample {args.sample_order}...")
    system_func, sample_param_distributions = load_function_from_log(
        args.log_path,
        args.sample_order,
    )

    # If the IC config has more state dimensions than the discovered system,
    # wrap the system to pad extra derivatives with zeros.
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
            # Detect whether the base system accepts an explicit time argument.
            base_accepts_time = False
            try:
                sig = inspect.signature(system_func)
                param_names = list(sig.parameters.keys())
                if len(param_names) >= 3 and param_names[2] in ('t', 'time', 'time_point'):
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
                f"Detected state/config mismatch: system has {inferred_n_states} "
                f"states but IC config for '{args.problem_name}' has {config_n_states}. "
                "Using padded wrapper so extra dimensions remain unused."
            )
            wrapped_system_func = padded_system

    param_distributions = sample_param_distributions
    if args.param_dist_path:
        with open(args.param_dist_path, 'r') as f:
            param_distributions = json.load(f)
        print(f"Loaded parameter distributions from {args.param_dist_path}")

    print(f"Simulating {args.n_patients} patient trajectories...")
    df, biomarker_names, t_eval, trajectories = simulate_trajectories(
        system_func=wrapped_system_func,
        problem_name=args.problem_name,
        n_patients=args.n_patients,
        param_distributions=param_distributions,
    )

    print(f"Biomarkers: {biomarker_names}")
    print(f"Time range: {t_eval[0]:.1f} - {t_eval[-1]:.1f} hours")
    print(f"Total observations: {len(df)}")
    print(64*'=')

    # Compare summary statistics: observed vs simulated.
    observed_data_path = f"data/{args.problem_name}/{args.problem_name}.csv"
    if os.path.exists(observed_data_path):
        print("\nComputing summary statistics comparison (observed vs simulated)...")
        observed_df = pd.read_csv(observed_data_path)
        std_params = compute_standardization_params(observed_df, biomarker_names)
        s_obs = compute_summary_stats_from_df(
            df=observed_df,
            biomarker_cols=biomarker_names,
            std_params=std_params,
            subject_col="id",
            time_col="t",
            verbose=False,
            standardization=True,
        )
        s_sim = compute_summary_stats(
            trajectories=trajectories,
            t_eval=t_eval,
            biomarker_names=biomarker_names,
            std_params=std_params,
            standardization=True,
        )
        stat_names = get_stat_names(biomarker_names)

        print(f"{'Idx':<4} {'Statistic':<40} {'Observed':>12} {'Simulated':>12} {'Diff':>12}")
        print("-" * 84)
        for idx, (name, so, ss) in enumerate(zip(stat_names, s_obs, s_sim)):
            print(f"{idx:<4} {name:<40} {so:12.4f} {ss:12.4f} {ss-so:12.4f}")

        euclidean_distance = float(np.linalg.norm(s_sim - s_obs))
        print("-" * 84)
        print(f"Euclidean distance (||s_sim - s_obs||_2): {euclidean_distance:.2f}")
    else:
        print(f"\nObserved data file not found at {observed_data_path}; skipping summary stats comparison.")

    if param_distributions is not None:
        quad_backend = "gpu" if getattr(wrapped_system_func, "_torch_backend", False) else "cpu"
        distance = evaluate_system_mntd(
            system_func=wrapped_system_func,
            problem_name=args.problem_name,
            param_distributions=param_distributions,
            verbose=True,
            sample_order=args.sample_order,
            backend=quad_backend,
            standardization=True,
        )
        if distance is not None:
            print(f"Euclidean summary-statistics distance: {distance:.2f}")
        else:
            print("Euclidean summary-statistics distance: None")
        print(64*'=')
    else:
        print("No parameter distributions provided; skipping quadratic score evaluation.")

    output_path = args.output
    if output_path is None and args.log_path:
        output_path = os.path.join(
            args.log_path,
            f'trajectories_sample_{args.sample_order}.png'
        )

    # Load problem-specific bin width from IC config.
    viz_bin_width = None
    try:
        ic_cfg = initial_condition_utils.load_ic_config(args.problem_name)
        viz_bin_width = initial_condition_utils.get_trajectory_bin_width(ic_cfg)
    except FileNotFoundError:
        pass

    # Optional fixed y-axis ranges for AKI visualizations.
    y_ranges = None
    if args.problem_name == 'aki' and biomarker_names == ['creatinine', 'bun', 'potassium']:
        y_ranges = {
            'trajectories': {
                'creatinine': (0.0, 11.0),
                'bun': (1.0, 120.0),
                'potassium': (2.0, 7.0),
            },
            'mean_ci': {
                'creatinine': (1.0, 3.0),
                'bun': (30.0, 50.0),
                'potassium': (3.8, 4.5),
            },
            'density': {
                'creatinine': (0.200, 19.100),
                'bun': (1.000, 160.000),
                'potassium': (3.000, 5.900),
            },
        }

    create_visualizations(df, biomarker_names, save_path=output_path, y_ranges=y_ranges, bin_width=viz_bin_width)

    print("Done!")


if __name__ == '__main__':
    main()
