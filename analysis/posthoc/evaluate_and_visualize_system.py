"""Visualize simulated ODE trajectories from discovered equations.

Usage:
    python analysis/evaluate_and_visualize_system.py --problem_name aki --sample_order 42 --log_path logs/aki_run1
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

from llmode.core import code_manipulation
from llmode.ode import initial_condition_utils
from llmode.ode import ode_simulator
from llmode.core import param_utils
from llmode.metrics.summary_stats import (
    compute_standardization_params,
    compute_summary_stats_from_df,
    compute_summary_stats,
    get_stat_names,
)
from llmode.metrics.euclidean_score import evaluate_system_euclidean_distance


def load_function_from_log(
    log_path: str,
    sample_order: int,
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

    print('Function:')
    print(function_str)
    print(64*'-')
    param_distributions = data.get('param_distributions')

    # Execute the function definition in an isolated namespace to obtain
    # a real Python callable `system` that we can pass to the ODE solver.
    namespace: dict = {"np": np}
    if is_torch_system:
        # Ensure that type annotations and torch operations inside the logged
        # function can be resolved when executing the definition.
        namespace["torch"] = torch
    exec(function_str, namespace)
    system_func = namespace.get("system")

    if callable(system_func):
        # Record whether this system is torch-based so that downstream helpers
        # can select the appropriate simulator backend.
        setattr(system_func, "_torch_backend", bool(is_torch_system))

    # Try to infer the number of state variables from the function body by
    # looking for assignments of the form:
    #     x1, x2, ... = biomarkers
    # Attach this as a private attribute so downstream helpers can detect
    # mismatches between the IC config dimension and the discovered system.
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
        # If we cannot parse the function body, skip annotation.
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
    # Load problem configuration
    config = initial_condition_utils.load_ic_config(problem_name)
    # Full biomarker order in the config (used for ICs / simulation).
    all_biomarker_names = initial_condition_utils.get_biomarker_order(config)
    # Subset of biomarkers that are actually observed in the data and used
    # for likelihood / summary statistics.
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    # Generate initial conditions
    ic_array, _ = initial_condition_utils.generate_initial_conditions(
        config,
        sample_size=n_patients,
        random_seed=config.get('random_seed', None),
    )

    # Sample parameters. Prefer LLM-inferred distributions when available;
    # otherwise fall back to a simple default.
    if param_distributions:
        params = param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n_patients,
            distribution="lognormal",
        )
    else:
        # Fallback: single-parameter vector of ones for all patients.
        # This should be overridden whenever param_distributions are present
        # in the sample log.
        params = np.ones((n_patients, 1))

    use_torch_backend = bool(getattr(system_func, "_torch_backend", False))

    # Simulate trajectories
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

    # Drop patients whose trajectories are numerically or physiologically invalid,
    # to match the behavior used in the quadratic-score evaluation.
    valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
        trajectories,
        config=config,
        check_nans=True,
    )
    valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
    print(f"Fraction of physiologically valid trajectories: {valid_fraction:.3f}")

    trajectories = trajectories[valid_mask]

    # Restrict trajectories to the subset of biomarkers that are observed in
    # the data, so that the biomarker dimension matches `biomarker_names`.
    observed_indices = [all_biomarker_names.index(name) for name in biomarker_names]
    trajectories_obs = trajectories[..., observed_indices]

    # Convert to DataFrame
    records = []
    n_valid = trajectories_obs.shape[0]
    for patient_id in range(n_valid):
        for t_idx, t in enumerate(t_eval):
            record = {
                'id': patient_id,
                't': t,
            }
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
):
    """Create comprehensive visualizations for simulated trajectories."""
    n_biomarkers = len(biomarker_names)
    fig = plt.figure(figsize=(18, 12))

    # Sample patients for individual trajectories
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
    time_bins = np.arange(0, df['t'].max() + 24, 24)
    df['time_bin'] = pd.cut(df['t'], bins=time_bins)

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, n_biomarkers + i + 1)
        # Explicitly set observed=False to retain current pandas behavior
        stats = df.groupby('time_bin', observed=False)[biomarker].agg(['mean', 'std', 'count'])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])

        # Use t-distribution for more accurate confidence intervals (correct for all sample sizes)
        # Calculate critical value from t-distribution based on degrees of freedom (n-1)
        stats['t_critical'] = stats['count'].apply(lambda n: scipy.stats.t.ppf(0.975, n-1))
        stats['ci'] = stats['t_critical'] * stats['se']

        # Original z-distribution approach (approximation valid for large samples):
        # stats['ci'] = 1.96 * stats['se']

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
    time_bins_fine = np.linspace(0, df['t'].max(), 30)
    colormaps = ['YlOrRd', 'YlGn', 'YlGnBu', 'PuRd', 'BuPu', 'OrRd']

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, 2 * n_biomarkers + i + 1)

        # Filter out NaN values
        valid_data = df[['t', biomarker]].dropna()

        if len(valid_data) > 0:
            # Use fixed y-axis range for density plots when provided;
            # otherwise fall back to the data-driven range.
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
                          extent=[0, df['t'].max(),
                                 y_min,
                                 y_max])
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
    parser.add_argument('--problem_name', type=str, required=True,
                       help='Problem name (e.g., aki)')
    parser.add_argument('--sample_order', type=int, required=True,
                       help='Sample order number from the log')
    parser.add_argument('--log_path', type=str, required=True,
                       help='Path to the log directory')
    parser.add_argument('--n_patients', type=int, default=100,
                       help='Number of patients to simulate (default: 100)')
    parser.add_argument('--param_dist_path', type=str, default=None,
                       help='Path to parameter distributions JSON file (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the visualization (default: show plot)')

    args = parser.parse_args()

    # Load the function from log
    print(f"Loading function from sample {args.sample_order}...")
    system_func, sample_param_distributions = load_function_from_log(
        args.log_path,
        args.sample_order,
    )

    # Detect the state dimension inferred from the logged function (if
    # available) and compare it to the IC config for this problem. For AKI we
    # may have runs with different numbers of state variables (e.g. 3 vs 5),
    # while the config can contain a superset of biomarkers. In that case we
    # wrap the system so it operates on the first `n_states` entries and returns
    # zero derivatives for any extra dimensions, keeping the observed biomarker
    # projections consistent.
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
            # Inspect whether the original system expects an explicit time
            # argument so we can delegate correctly inside the wrapper.
            base_accepts_time = False
            try:
                sig = inspect.signature(system_func)
                param_names = list(sig.parameters.keys())
                if len(param_names) >= 3 and param_names[2] in ('t', 'time', 'time_point'):
                    base_accepts_time = True
            except (TypeError, ValueError):
                base_accepts_time = False

            def padded_system(
                biomarkers: np.ndarray,
                params: np.ndarray,
                t: float | None = None,
                _base_func: Callable = system_func,
                _accepts_time: bool = base_accepts_time,
                _n_states: int = inferred_n_states,
            ) -> np.ndarray:
                # Use only the first `_n_states` entries for the discovered
                # system and pad any extra dimensions with zeros so the shape
                # matches the IC config.
                core = biomarkers[:_n_states]
                if _accepts_time:
                    core_deriv = _base_func(core, params, t)
                else:
                    core_deriv = _base_func(core, params)
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

    # Load parameter distributions:
    #   - if a JSON path is provided, use that;
    #   - otherwise fall back to any distributions stored in the sample log.
    param_distributions = sample_param_distributions
    if args.param_dist_path:
        with open(args.param_dist_path, 'r') as f:
            param_distributions = json.load(f)
        print(f"Loaded parameter distributions from {args.param_dist_path}")

    # Simulate trajectories
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
            diff = ss - so
            print(f"{idx:<4} {name:<40} {so:12.4f} {ss:12.4f} {diff:12.4f}")

        # Euclidean distance between observed and simulated summary-statistics vectors.
        diffs = s_sim - s_obs
        euclidean_distance = float(np.linalg.norm(diffs))
        print("-" * 84)
        print(f"Euclidean distance (||s_sim - s_obs||_2): {euclidean_distance:.4f}")
    else:
        print(f"\nObserved data file not found at {observed_data_path}; skipping summary stats comparison.")

    # Compute Euclidean summary-statistics distance (if parameter priors are available).
    # For torch-based systems, ensure we use the torch simulator backend so that
    # the trajectories used for scoring are consistent with those used for
    # visualization and centralized evaluation.
    if param_distributions is not None:
        quad_backend = "gpu" if getattr(wrapped_system_func, "_torch_backend", False) else "cpu"
        distance = evaluate_system_euclidean_distance(
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

    # Create visualizations
    # print("Creating visualizations...")
    output_path = args.output
    if output_path is None and args.log_path:
        output_path = os.path.join(
            args.log_path,
            f'trajectories_sample_{args.sample_order}.png'
        )

    # Optional fixed y-axis ranges for AKI problem visualizations.
    y_ranges = None
    if args.problem_name == 'aki' and biomarker_names == ['creatinine', 'bun', 'potassium']:
        y_ranges = {
            'trajectories': {
                'creatinine': (0.0, 11.0), # (0.340, 10.460),
                'bun': (1.0, 120.0), # (3.900, 116.100),
                'potassium': (2.0, 7.0), # (3.300, 5.500),
            },
            'mean_ci': {
                'creatinine': (1.0, 3.0), # (1.631, 2.773),
                'bun': (30.0, 50.0), # (31.914, 41.068),
                'potassium': (3.8, 4.5), # (4.021, 4.322),
            },
            'density': {
                'creatinine': (0.200, 19.100),
                'bun': (1.000, 160.000),
                'potassium': (3.000, 5.900),
            },
        }

    create_visualizations(df, biomarker_names, save_path=output_path, y_ranges=y_ranges)

    print("Done!")


if __name__ == '__main__':
    main()
