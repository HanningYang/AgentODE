"""Synthetic likelihood evaluation utilities for ODE systems.

Implements summary statistics and Wood's synthetic likelihood:

    log L(theta) = -1/2 [log|Sigma_hat| + (s_obs - mu_hat)^T Sigma_hat^{-1} (s_obs - mu_hat)]

This is designed to be problem-agnostic. The caller provides:
  - Observed data (as a DataFrame) and which biomarker columns to use.
  - A simulator callback that, given (n_patients, param_sampler, t_eval),
    returns trajectories of shape (n_patients, n_timepoints, n_biomarkers).

The helper `evaluate_system_logsl` wires this up for a given problem using
the repository's initial-condition utilities, simulators, and parameter priors.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


N_PATIENTS_DEFAULT = 100
N_SIMULATIONS_DEFAULT = 100
REGULARIZATION_DEFAULT = 1e-6


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
        params[col] = {'mean': mean, 'std': std}
    return params


def standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize values using given mean and std."""
    return (values - mean) / std


def compute_summary_stats(
    trajectories: np.ndarray,
    t_eval: np.ndarray,
    biomarker_names: List[str],
    std_params: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Compute summary statistics from simulated trajectories.

    Stats per biomarker:
      - 2: Quantile polynomial deg0, deg1 on standardized values
      - 1: Lag-1 autocorrelation (across time)

    Cross-biomarker stats:
      - For each pair (b1, b2) with b2 > b1: correlation of first differences.

    Total stats:
      3 * n_biomarkers + n_biomarkers * (n_biomarkers - 1) / 2.
    """
    n_patients, n_times, n_bio = trajectories.shape
    assert n_bio == len(biomarker_names)

    stats: List[float] = []

    # 1. Marginal quantile polynomials on standardized values (2 * n_bio).
    for b, bio_name in enumerate(biomarker_names):
        all_values = trajectories[:, :, b].ravel()
        all_values = all_values[~np.isnan(all_values)]

        standardized_values = standardize(
            all_values,
            std_params[bio_name]['mean'],
            std_params[bio_name]['std'],
        )
        coeffs = polynomial_quantile_stats_deg01(standardized_values)
        stats.extend(coeffs.tolist())

    # 2. Temporal: lag-1 autocorrelation (n_bio).
    for b, _bio_name in enumerate(biomarker_names):
        y_t: List[float] = []
        y_t1: List[float] = []
        for i in range(n_patients):
            vals = trajectories[i, :, b]
            if not np.any(np.isnan(vals)) and vals.size > 1:
                y_t.extend(vals[1:].tolist())
                y_t1.extend(vals[:-1].tolist())

        if len(y_t) > 1:
            autocorr, _ = spearmanr(y_t, y_t1)
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        stats.append(autocorr)

    # 3. Dynamic: difference correlations between biomarker pairs.
    for b1 in range(n_bio):
        for b2 in range(b1 + 1, n_bio):
            diffs1: List[float] = []
            diffs2: List[float] = []
            for i in range(n_patients):
                v1 = trajectories[i, :, b1]
                v2 = trajectories[i, :, b2]
                if (
                    not np.any(np.isnan(v1))
                    and not np.any(np.isnan(v2))
                    and v1.size > 1
                    and v2.size > 1
                ):
                    dv1 = np.diff(v1)
                    dv2 = np.diff(v2)
                    m = min(dv1.size, dv2.size)
                    diffs1.extend(dv1[:m].tolist())
                    diffs2.extend(dv2[:m].tolist())

            if len(diffs1) > 1:
                corr, _ = spearmanr(diffs1, diffs2)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            stats.append(corr)

    return np.asarray(stats, dtype=float)


def compute_summary_stats_from_df(
    df: pd.DataFrame,
    biomarker_cols: List[str],
    std_params: Dict[str, Dict[str, float]],
    subject_col: str = 'subject_id',
    episode_col: str = 'hadm_id',
    time_col: str = 'hours_from_admission',
    verbose: bool = False,
) -> np.ndarray:
    """Compute summary statistics from observed longitudinal data."""
    stats: List[float] = []

    if verbose:
        print("\n" + "="*80)
        print("COMPUTING SUMMARY STATISTICS FROM OBSERVED DATA")
        print("="*80)
        print(f"\nDataset shape: {df.shape}")
        print(f"Unique subjects: {df[subject_col].nunique()}")
        print(f"Unique episodes: {df[episode_col].nunique()}")
        print(f"\nBiomarker columns: {biomarker_cols}")

    # 1. Marginal quantile polynomials on standardized values.
    if verbose:
        print("\n[1] Quantile Polynomial Coefficients (deg0, deg1):")
    for col in biomarker_cols:
        values = df[col].dropna().to_numpy()
        if verbose:
            print(f"\n  {col}:")
            print(f"    Raw - Count: {len(values)}, Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
            print(f"    Standardization - Mean: {std_params[col]['mean']:.4f}, Std: {std_params[col]['std']:.4f}")
        standardized_values = standardize(
            values,
            std_params[col]['mean'],
            std_params[col]['std'],
        )
        coeffs = polynomial_quantile_stats_deg01(standardized_values)
        if verbose:
            print(f"    Quantile poly deg0: {coeffs[0]:.6f}, deg1: {coeffs[1]:.6f}")
        stats.extend(coeffs.tolist())

    # 2. Temporal: lag-1 autocorrelation per biomarker across episodes.
    if verbose:
        print("\n[2] Lag-1 Autocorrelation:")
    grouped = df.groupby([subject_col, episode_col])
    for col in biomarker_cols:
        y_t: List[float] = []
        y_t1: List[float] = []
        for _, group in grouped:
            g = group.sort_values(time_col)
            vals = g[col].dropna().to_numpy()
            if vals.size > 1:
                y_t.extend(vals[1:].tolist())
                y_t1.extend(vals[:-1].tolist())

        if len(y_t) > 1:
            autocorr, _ = spearmanr(y_t, y_t1)
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        if verbose:
            print(f"  {col}: {autocorr:.6f} (from {len(y_t)} lag pairs)")
        stats.append(autocorr)

    # 3. Dynamic: difference correlations between biomarker pairs.
    if verbose:
        print("\n[3] Difference Correlations (between biomarker pairs):")
    for i, col1 in enumerate(biomarker_cols):
        for j, col2 in enumerate(biomarker_cols):
            if j <= i:
                continue
            diffs1: List[float] = []
            diffs2: List[float] = []
            for _, group in grouped:
                g = group.sort_values(time_col)
                v1 = g[col1].dropna().to_numpy()
                v2 = g[col2].dropna().to_numpy()
                m = min(v1.size, v2.size)
                if m > 1:
                    dv1 = np.diff(v1[:m])
                    dv2 = np.diff(v2[:m])
                    diffs1.extend(dv1.tolist())
                    diffs2.extend(dv2.tolist())

            if len(diffs1) > 1:
                corr, _ = spearmanr(diffs1, diffs2)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            if verbose:
                print(f"  {col1} vs {col2}: {corr:.6f} (from {len(diffs1)} diff pairs)")
            stats.append(corr)

    result = np.asarray(stats, dtype=float)

    if verbose:
        stat_names = get_stat_names(biomarker_cols)
        print("\n" + "-"*80)
        print(f"SUMMARY STATISTICS VECTOR (s_obs) - Total: {len(result)}")
        print("-"*80)
        print(f"{'Index':<8} {'Statistic Name':<40} {'Value':>12}")
        print("-"*80)
        for idx, (name, value) in enumerate(zip(stat_names, result)):
            print(f"{idx:<8} {name:<40} {value:12.6f}")
        print("="*80 + "\n")

    return result


def get_stat_names(biomarker_cols: List[str]) -> List[str]:
    """Return human-readable names for summary statistics."""
    names: List[str] = []

    # Quantile polynomial coefficients.
    for col in biomarker_cols:
        names.append(f"{col}_std_quantile_deg0")
        names.append(f"{col}_std_quantile_deg1")

    # Lag-1 autocorrelation.
    for col in biomarker_cols:
        names.append(f"{col}_lag1_autocorr")

    # Difference correlations.
    for i, col1 in enumerate(biomarker_cols):
        for j, col2 in enumerate(biomarker_cols):
            if j <= i:
                continue
            names.append(f"diff_corr_{col1[:2]}_{col2[:2]}")

    return names


class LogSyntheticLikelihood:
    """Wood's synthetic likelihood for ODE-based simulators.

    The caller provides:
      - Observed summary statistics `s_obs`
      - A simulator callback:
            simulator(n_patients, param_sampler, t_eval) -> trajectories
      - A summary-statistic function for simulated trajectories
            summary_fn(trajectories, t_eval, biomarker_names, std_params)
    """

    def __init__(
        self,
        s_obs: np.ndarray,
        stat_names: List[str],
        std_params: Dict[str, Dict[str, float]],
        biomarker_names: List[str],
        simulator: Callable[[int, Callable[[int], np.ndarray], np.ndarray], np.ndarray],
        n_patients: int = N_PATIENTS_DEFAULT,
        n_simulations: int = N_SIMULATIONS_DEFAULT,
        regularization: float = REGULARIZATION_DEFAULT,
    ):
        self.s_obs = s_obs
        self.stat_names = stat_names
        self.std_params = std_params
        self.biomarker_names = biomarker_names
        self.simulator = simulator
        self.n_patients = n_patients
        self.n_simulations = n_simulations
        self.regularization = regularization
        self.n_stats = int(s_obs.size)

    def run_simulations(
        self,
        param_sampler: Callable[[int], np.ndarray],
        t_eval: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run `n_simulations` simulations and estimate mu_hat and Sigma_hat."""
        simulated_stats: List[np.ndarray] = []

        for sim_idx in range(self.n_simulations):
            if verbose and (sim_idx + 1) % 50 == 0:
                print(f"  Simulation {sim_idx + 1}/{self.n_simulations}")

            trajectories = self.simulator(self.n_patients, param_sampler, t_eval)
            s_sim = compute_summary_stats(
                trajectories,
                t_eval,
                self.biomarker_names,
                self.std_params,
            )
            simulated_stats.append(s_sim)

        simulated_stats_arr = np.vstack(simulated_stats)

        mu_hat = np.mean(simulated_stats_arr, axis=0)
        sigma_hat = np.cov(simulated_stats_arr, rowvar=False)

        if sigma_hat.ndim == 0:
            sigma_hat = np.array([[float(sigma_hat)]], dtype=float)

        # Regularize the covariance for numerical stability.
        sigma_hat = sigma_hat + self.regularization * np.eye(sigma_hat.shape[0], dtype=float)

        return mu_hat, sigma_hat, simulated_stats_arr

    def compute_log_likelihood(
        self,
        mu_hat: np.ndarray,
        sigma_hat: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute log synthetic likelihood and diagnostic details."""
        residual = self.s_obs - mu_hat

        # Log determinant term.
        sign, log_det = np.linalg.slogdet(sigma_hat)
        if sign <= 0:
            return float('-inf'), {'error': np.array(['Sigma not positive definite'])}

        # Inverse and quadratic form.
        try:
            sigma_inv = np.linalg.inv(sigma_hat)
            quadratic = float(residual @ sigma_inv @ residual)
        except np.linalg.LinAlgError:
            return float('-inf'), {'error': np.array(['Sigma inversion failed'])}

        log_likelihood = -0.5 * (log_det + quadratic)

        per_stat_contrib = -0.5 * residual**2 * np.diag(sigma_inv)

        details: Dict[str, np.ndarray] = {
            'log_det_term': np.array([-0.5 * log_det]),
            'quadratic_term': np.array([-0.5 * quadratic]),
            'residual': residual,
            'residual_norm': np.array([np.linalg.norm(residual)]),
            'mu_hat': mu_hat,
            'sigma_hat': sigma_hat,
            'sigma_inv_diag': np.diag(sigma_inv),
            'per_stat_contrib': per_stat_contrib,
            'sigma_condition_number': np.array([np.linalg.cond(sigma_hat)]),
        }
        return float(log_likelihood), details

    def evaluate(
        self,
        param_sampler: Callable[[int], np.ndarray],
        t_eval: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Full evaluation: run simulations and compute log-likelihood."""
        mu_hat, sigma_hat, sim_stats = self.run_simulations(
            param_sampler,
            t_eval,
            verbose,
        )
        log_likelihood, details = self.compute_log_likelihood(mu_hat, sigma_hat)
        details['simulated_stats'] = sim_stats
        return log_likelihood, details


def evaluate_system_logsl(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
) -> float | None:
    """Evaluate a system function via log synthetic likelihood for a given problem.

    This helper wires together:
      - initial-condition generation from the problem's IC config,
      - observed data loading from `data/{problem_name}/{problem_name}.csv`,
      - parameter sampling from LLM-inferred priors when available,
      - ODE simulation using `ode_simulator.simulate_ode_system`,
      - synthetic likelihood computation against observed summary statistics.
    """
    from llmode import initial_condition_utils
    from llmode import ode_simulator
    from llmode import param_utils

    # Load configuration and observed data.
    config = initial_condition_utils.load_ic_config(problem_name)
    biomarker_names = initial_condition_utils.get_observed_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    observed_data_path = f'data/{problem_name}/{problem_name}.csv'
    observed_df = pd.read_csv(observed_data_path)

    # Standardization parameters and observed stats.
    std_params = compute_standardization_params(observed_df, biomarker_names)
    s_obs = compute_summary_stats_from_df(
        observed_df,
        biomarker_names,
        std_params,
        subject_col='subject_id',
        episode_col='hadm_id',
        time_col='hours_from_admission',
    )
    stat_names = get_stat_names(biomarker_names)

    # If no parameter distributions are available, we cannot form a meaningful
    # synthetic likelihood; treat this as unevaluable.
    if param_distributions is None:
        return None

    def param_sampler(n: int) -> np.ndarray:
        return param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n,
            distribution="lognormal",
        )

    def simulator(
        n_patients: int,
        sampler_fn: Callable[[int], np.ndarray],
        t_grid: np.ndarray,
    ) -> np.ndarray:
        ic_array, _ = initial_condition_utils.generate_initial_conditions(
            config,
            sample_size=n_patients,
            random_seed=config.get('random_seed', None),
        )
        param_sets = sampler_fn(n_patients)
        return ode_simulator.simulate_ode_system(
            system_func=system_func,
            initial_conditions=ic_array,
            time_grid=t_grid,
            params=param_sets,
            method='odeint',
        )

    log_sl = LogSyntheticLikelihood(
        s_obs=s_obs,
        stat_names=stat_names,
        std_params=std_params,
        biomarker_names=biomarker_names,
        simulator=simulator,
        n_patients=N_PATIENTS_DEFAULT,
        n_simulations=N_SIMULATIONS_DEFAULT,
        regularization=REGULARIZATION_DEFAULT,
    )

    log_likelihood, _details = log_sl.evaluate(param_sampler, t_eval, verbose=False)
    if not np.isfinite(log_likelihood):
        return None
    return float(log_likelihood)
