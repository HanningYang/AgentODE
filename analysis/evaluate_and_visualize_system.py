"""Visualize simulated ODE trajectories from discovered equations.

Usage:
    python analysis/evaluate_and_visualize_system.py --problem_name aki --sample_order 42 --log_path logs/aki_run1
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llmode import code_manipulation
from llmode import initial_condition_utils
from llmode import ode_simulator
from llmode import param_utils
from llmode.synthetic_likelihood import (
    evaluate_system_logsl,
    compute_standardization_params,
    compute_summary_stats_from_df,
    compute_summary_stats,
    get_stat_names,
)


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
    print('Function:')
    print(function_str)
    print(64*'-')
    param_distributions = data.get('param_distributions')

    # Execute the function definition in an isolated namespace to obtain
    # a real Python callable `system` that we can pass to the ODE solver.
    namespace: dict = {"np": np}
    exec(function_str, namespace)
    system_func = namespace.get("system")

    if not callable(system_func):
        raise ValueError(
            "Loaded function is not callable. Expected a definition like "
            "`def system(biomarkers, params): ...` in the log."
        )

    return system_func, param_distributions


def simulate_trajectories(
    system_func: Callable,
    problem_name: str,
    n_patients: int = 100,
    param_distributions: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Simulate patient trajectories using the discovered ODE system.

    Returns:
        df: DataFrame with columns [subject_id, hadm_id, hours_from_admission, biomarker1, biomarker2, ...]
        biomarker_names: List of biomarker column names
        t_eval: Time grid used for simulation
    """
    # Load problem configuration
    config = initial_condition_utils.load_ic_config(problem_name)
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

    # Simulate trajectories
    trajectories = ode_simulator.simulate_ode_system(
        system_func=system_func,
        initial_conditions=ic_array,
        time_grid=t_eval,
        params=params,
        method='odeint',
    )

    # Convert to DataFrame
    records = []
    for patient_id in range(n_patients):
        for t_idx, t in enumerate(t_eval):
            record = {
                'subject_id': patient_id,
                'hadm_id': patient_id,  # One episode per patient
                'hours_from_admission': t,
            }
            for bio_idx, bio_name in enumerate(biomarker_names):
                record[bio_name] = trajectories[patient_id, t_idx, bio_idx]
            records.append(record)

    df = pd.DataFrame(records)
    return df, biomarker_names, t_eval, trajectories


def create_visualizations(
    df: pd.DataFrame,
    biomarker_names: list[str],
    save_path: str = None,
):
    """Create comprehensive visualizations for simulated trajectories."""
    n_biomarkers = len(biomarker_names)
    fig = plt.figure(figsize=(18, 12))

    # Sample patients for individual trajectories
    sample_patients = df['subject_id'].unique()[:5]

    # Row 1: Individual patient trajectories
    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, i + 1)
        for patient_id in sample_patients:
            patient_data = df[df['subject_id'] == patient_id]
            ax.plot(patient_data['hours_from_admission'], patient_data[biomarker],
                   marker='o', alpha=0.6, label=f'Patient {patient_id}')
        ax.set_xlabel('Hours from Admission')
        ax.set_ylabel(biomarker)
        ax.set_title(f'{biomarker} Trajectories (Sample)')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Aggregate temporal trends with confidence intervals
    time_bins = np.arange(0, df['hours_from_admission'].max() + 24, 24)
    df['time_bin'] = pd.cut(df['hours_from_admission'], bins=time_bins)

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, n_biomarkers + i + 1)
        # Explicitly set observed=False to retain current pandas behavior
        stats = df.groupby('time_bin', observed=False)[biomarker].agg(['mean', 'std', 'count'])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        stats['ci'] = 1.96 * stats['se']
        bin_centers = [(interval.left + interval.right) / 2 for interval in stats.index]

        ax.plot(bin_centers, stats['mean'], linewidth=2, label='Mean')
        ax.fill_between(bin_centers,
                        stats['mean'] - stats['ci'],
                        stats['mean'] + stats['ci'],
                        alpha=0.3, label='95% CI')
        ax.set_xlabel('Hours from Admission')
        ax.set_ylabel(biomarker)
        ax.set_title(f'{biomarker}: Mean ± 95% CI')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 3: Distribution density heatmaps
    time_bins_fine = np.linspace(0, df['hours_from_admission'].max(), 30)
    colormaps = ['YlOrRd', 'YlGn', 'YlGnBu', 'PuRd', 'BuPu', 'OrRd']

    for i, biomarker in enumerate(biomarker_names):
        ax = plt.subplot(3, n_biomarkers, 2 * n_biomarkers + i + 1)

        # Filter out NaN values
        valid_data = df[['hours_from_admission', biomarker]].dropna()

        if len(valid_data) > 0:
            bio_bins = np.linspace(valid_data[biomarker].min(),
                                  valid_data[biomarker].max(), 30)
            H, xedges, yedges = np.histogram2d(
                valid_data['hours_from_admission'],
                valid_data[biomarker],
                bins=[time_bins_fine, bio_bins]
            )

            im = ax.imshow(H.T, origin='lower', aspect='auto',
                          cmap=colormaps[i % len(colormaps)],
                          extent=[0, df['hours_from_admission'].max(),
                                 valid_data[biomarker].min(),
                                 valid_data[biomarker].max()])
            ax.set_xlabel('Hours from Admission')
            ax.set_ylabel(biomarker)
            ax.set_title(f'{biomarker} Density over Time')
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
        system_func=system_func,
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
            observed_df,
            biomarker_cols=biomarker_names,
            std_params=std_params,
            subject_col="subject_id",
            episode_col="hadm_id",
            time_col="hours_from_admission",
            verbose=False,
        )
        s_sim = compute_summary_stats(
            trajectories=trajectories,
            t_eval=t_eval,
            biomarker_names=biomarker_names,
            std_params=std_params,
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

    # Compute synthetic log-likelihood (if parameter priors are available)
    if param_distributions is not None:
        # print("Evaluating synthetic log-likelihood (LogSL)...")
        log_sl = evaluate_system_logsl(
            system_func=system_func,
            problem_name=args.problem_name,
            param_distributions=param_distributions,
            verbose=True,
            sample_order=args.sample_order,
        )
        print(f"Synthetic Log-Likelihood (LogSL): {log_sl}")
        print(64*'=')
    else:
        print("No parameter distributions provided; skipping synthetic likelihood.")

    # Create visualizations
    # print("Creating visualizations...")
    output_path = args.output
    if output_path is None and args.log_path:
        output_path = os.path.join(
            args.log_path,
            f'trajectories_sample_{args.sample_order}.png'
        )

    create_visualizations(df, biomarker_names, save_path=output_path)

    print("Done!")


if __name__ == '__main__':
    main()
