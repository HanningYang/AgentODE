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
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from llmode.summary_stats import (
    polynomial_quantile_stats_deg01,
    compute_standardization_params,
    standardize,
    compute_summary_stats,
    compute_summary_stats_from_df,
    get_stat_names,
    get_observed_summary,
)


N_PATIENTS_DEFAULT = 100
N_SIMULATIONS_DEFAULT = 150
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
      - 1: Population-level trend (Spearman correlation between time and
           population mean trajectory)

    Cross-biomarker stats:
      - For each pair (b1, b2) with b2 > b1: correlation of first differences.

    Total stats:
      4 * n_biomarkers + n_biomarkers * (n_biomarkers - 1) / 2.
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

    # 3. Population-level trend: Spearman correlation between time and
    #    population mean trajectory for each biomarker (n_bio).
    for b, _bio_name in enumerate(biomarker_names):
        mean_traj: List[float] = []
        for t_idx in range(n_times):
            vals = trajectories[:, t_idx, b]
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                mean_traj.append(float(np.mean(vals)))
            else:
                mean_traj.append(np.nan)

        mean_traj_arr = np.asarray(mean_traj, dtype=float)
        time_vec = np.asarray(t_eval, dtype=float)
        mask = ~np.isnan(mean_traj_arr)
        mean_traj_masked = mean_traj_arr[mask]
        time_vec_masked = time_vec[mask]

        if mean_traj_masked.size > 1:
            trend, _ = spearmanr(time_vec_masked, mean_traj_masked)
            if np.isnan(trend):
                trend = 0.0
        else:
            trend = 0.0
        stats.append(float(trend))

    # 4. Dynamic: difference correlations between biomarker pairs.
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

    # 3. Population-level trend: Spearman correlation between time and
    #    population mean trajectory for each biomarker.
    if verbose:
        print("\n[3] Population-level trend (Spearman time vs mean biomarker):")
    for col in biomarker_cols:
        sub = df[[time_col, col]].dropna(subset=[col])
        if sub.empty:
            trend = 0.0
            n_timepoints = 0
        else:
            grouped_time = sub.groupby(time_col)[col].mean()
            time_vec = grouped_time.index.to_numpy(dtype=float)
            mean_traj = grouped_time.to_numpy(dtype=float)
            n_timepoints = mean_traj.size
            if n_timepoints > 1:
                trend, _ = spearmanr(time_vec, mean_traj)
                if np.isnan(trend):
                    trend = 0.0
            else:
                trend = 0.0
        if verbose:
            print(f"  {col}: {trend:.6f} (from {n_timepoints} time points)")
        stats.append(float(trend))

    # 4. Dynamic: difference correlations between biomarker pairs.
    if verbose:
        print("\n[4] Difference Correlations (between biomarker pairs):")
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

    # Population-level trend.
    for col in biomarker_cols:
        names.append(f"{col}_pop_trend_spearman")

    # Difference correlations.
    for i, col1 in enumerate(biomarker_cols):
        for j, col2 in enumerate(biomarker_cols):
            if j <= i:
                continue
            names.append(f"diff_corr_{col1[:2]}_{col2[:2]}")

    return names


def _human_readable_stat_label(stat_name: str, biomarker_names: List[str]) -> str:
    """Map internal stat name to a human-readable label with biomarker context."""
    # Per-biomarker stats: "<bio>_<kind>"
    for bio in biomarker_names:
        prefix = f"{bio}_"
        if stat_name.startswith(prefix):
            kind = stat_name[len(prefix):]
            if kind == "std_quantile_deg0":
                label = "Baseline / intercept ↔ quantile deg0 (location)" # "Baseline Level (Intercept)"
            elif kind == "std_quantile_deg1":
                label = "Spread / variability ↔ marginal quantile slope (deg1) / dispersion" # "Spread / Variability"
            elif kind == "lag1_autocorr":
                label = "Temporal smoothness ↔ lag-1 Spearman autocorr" # "Temporal Smoothness"
            elif kind == "pop_trend_spearman":
                label = "Population drift ↔ time–mean Spearman trend" # "Overall Population Drift"
            else:
                label = kind
            return f"{bio}: {label}"

    # Cross-biomarker dynamics: "diff_corr_{b1}_{b2}" where b1/b2 are prefixes.
    if stat_name.startswith("diff_corr_"):
        parts = stat_name.split("_")
        if len(parts) == 4:
            _, _, p1, p2 = parts
            prefix_map = {bio[:2]: bio for bio in biomarker_names}
            bio1 = prefix_map.get(p1, p1)
            bio2 = prefix_map.get(p2, p2)
            return f"{bio1} - {bio2}: Dynamic coupling ↔ first-difference cross-biomarker Spearman"

    return stat_name


def generate_llm_feedback(
    log_likelihood: float,
    s_obs: np.ndarray,
    details: Dict[str, np.ndarray],
    stat_names: List[str],
    biomarker_names: List[str],
) -> str:
    """Generate aggregated feedback string for guiding LLM optimization."""

    mu_hat = details['mu_hat']
    sigma_hat = details['sigma_hat']
    per_stat_contrib = details['per_stat_contrib']
    residual = details['residual']
    sim_std = np.sqrt(np.diag(sigma_hat))  # Simulation std per stat

    # ========================================
    # 1. Per-Biomarker Summary
    # ========================================
    biomarker_summary: Dict[str, Dict[str, float | str]] = {}

    for bio in biomarker_names:
        marginal_idx = [
            i for i, n in enumerate(stat_names)
            if n.startswith(f"{bio}_") and ('std_quantile_deg0' in n or 'std_quantile_deg1' in n)
        ]
        temporal_idx = [
            i for i, n in enumerate(stat_names)
            if n.startswith(f"{bio}_") and ('lag1_autocorr' in n or 'pop_trend_spearman' in n)
        ]

        marginal_contrib = float(sum(per_stat_contrib[i] for i in marginal_idx)) if marginal_idx else 0.0
        temporal_contrib = float(sum(per_stat_contrib[i] for i in temporal_idx)) if temporal_idx else 0.0
        total = marginal_contrib + temporal_contrib

        biomarker_summary[bio] = {
            'marginal': marginal_contrib,
            'temporal': temporal_contrib,
            'total': total,
        }

    # Assign priority (more negative total -> higher priority).
    sorted_bios = sorted(biomarker_summary.items(), key=lambda x: x[1]['total'])
    for rank, (bio, _) in enumerate(sorted_bios):
        if rank == 0:
            biomarker_summary[bio]['priority'] = 'Critical'
        elif rank == 1:
            biomarker_summary[bio]['priority'] = 'High'
        else:
            biomarker_summary[bio]['priority'] = 'Medium'

    # ========================================
    # 2. Cross-Biomarker Dynamics
    # ========================================
    corr_stats = []
    prefix_map = {bio[:2]: bio for bio in biomarker_names}
    for i, name in enumerate(stat_names):
        if name.startswith("diff_corr_"):
            parts = name.split("_")
            if len(parts) == 4:
                _, _, p1, p2 = parts
                bio1 = prefix_map.get(p1, p1)
                bio2 = prefix_map.get(p2, p2)
                pair_label = f"{bio1} - {bio2}"
            else:
                pair_label = name

            corr_stats.append({
                'pair': pair_label,
                'name': name,
                'observed': float(s_obs[i]),
                'simulated': float(mu_hat[i]),
                'sim_std': float(sim_std[i]),
                'contribution': float(per_stat_contrib[i]),
                'gap': float(mu_hat[i] - s_obs[i]),
            })

    # ========================================
    # 3. Top Problems (sorted by contribution)
    # ========================================
    # problems = []
    # for i, name in enumerate(stat_names):
    #     gap = float(mu_hat[i] - s_obs[i])

    #     if 'std_quantile_deg0' in name:
    #         direction = "values too HIGH" if gap > 0 else "values too LOW"
    #     elif 'std_quantile_deg1' in name:
    #         direction = "spread too WIDE" if gap > 0 else "spread too NARROW"
    #     elif 'lag1_autocorr' in name:
    #         direction = "too persistent" if gap > 0 else "too variable"
    #     elif 'pop_trend_spearman' in name:
    #         direction = "trend too strong" if gap > 0 else "trend too weak"
    #     elif 'diff_corr' in name:
    #         direction = "too coupled" if gap > 0 else "too independent"
    #     else:
    #         direction = "mismatch"

    problems = []
    for i, name in enumerate(stat_names):
        obs = float(s_obs[i])
        sim = float(mu_hat[i])
        gap = sim - obs  # simulated - observed

        if 'std_quantile_deg0' in name:
            # Location shift of standardized marginal distribution
            direction = "values too HIGH" if gap > 0 else "values too LOW"

        elif 'std_quantile_deg1' in name:
            # Slope/scale of quantile function ~ dispersion of standardized values
            direction = "spread too WIDE" if gap > 0 else "spread too NARROW"

        elif 'lag1_autocorr' in name:
            # Higher lag-1 => more persistence/smoothness
            direction = "too persistent" if gap > 0 else "too variable"

        elif 'pop_trend_spearman' in name:
            # Option A: compare magnitude for "strength" and track sign mismatch
            # Use a small threshold to avoid overreacting to near-zero observed trends
            if np.sign(sim) != np.sign(obs) and abs(obs) > 0.10:
                direction = "wrong direction"
            elif abs(sim) > abs(obs):
                direction = "trend too strong"
            else:
                direction = "trend too weak"

        elif 'diff_corr' in name:
            # Cross-biomarker dynamic coupling via first-difference correlation
            # If sign flips (and observed isn't near zero), call it out explicitly
            if np.sign(sim) != np.sign(obs) and abs(obs) > 0.10:
                direction = "coupling sign mismatch"
            elif abs(sim) > abs(obs):
                direction = "too coupled"
            else:
                direction = "too independent"

        else:
            direction = "mismatch"


        readable_name = _human_readable_stat_label(name, biomarker_names)

        problems.append({
            'name': readable_name,
            'observed': float(s_obs[i]),
            'simulated': float(mu_hat[i]),
            'sim_std': float(sim_std[i]),
            'contribution': float(per_stat_contrib[i]),
            'direction': direction,
        })

    problems.sort(key=lambda x: x['contribution'])

    # ========================================
    # 4. Format Output
    # ========================================
    output_lines: List[str] = []
    output_lines.append(f"## Evaluation Score: {log_likelihood:.2f} (higher is better)\n")

    # Per-biomarker table (disabled for now; kept for future use).
    # output_lines.append("## Per-Biomarker Summary:\n")
    # output_lines.append("| Biomarker | Marginal | Temporal | Total | Priority |")
    # output_lines.append("|-----------|----------|----------|-------|----------|")
    # for bio in biomarker_names:
    #     data = biomarker_summary[bio]
    #     output_lines.append(
    #         f\"| {bio} | {data['marginal']:.1f} | {data['temporal']:.1f} | "
    #         f\"{data['total']:.1f} | {data['priority']} |\"
    #     )

    # Cross-biomarker table (disabled for now; kept for future use).
    # output_lines.append("\n## Cross-Biomarker Dynamics (spearman correlation):")
    # output_lines.append("| Pair | Observed | Simulated | Sim Std | Contribution | Gap |")
    # output_lines.append("|------|----------|-----------|---------|--------------|-----|")
    # for cs in corr_stats:
    #     output_lines.append(
    #         f\"| {cs['pair']} | {cs['observed']:.3f} | {cs['simulated']:.3f} | "
    #         f\"{cs['sim_std']:.3f} | {cs['contribution']:.2f} | {cs['gap']:+.3f} |\"
    #     )

    # Top 5 problems
    output_lines.append("\n## Top 5 Problems:")
    for i, prob in enumerate(problems[:5]):
        output_lines.append(f"{i+1}. **{prob['name']}**: {prob['direction']}")
        output_lines.append(
            f"   - Observed: {prob['observed']:.3f}, "
            f"Simulated: {prob['simulated']:.3f} (±{prob['sim_std']:.3f})"
        )
        output_lines.append(f"   - Contribution: {prob['contribution']:.2f}")

    return "\n".join(output_lines) + "\n"


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
        n_jobs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run `n_simulations` simulations and estimate mu_hat and Sigma_hat."""
        def _one_sim(sim_idx: int) -> np.ndarray:
            if verbose and (sim_idx + 1) % 50 == 0:
                print(f"  Simulation {sim_idx + 1}/{self.n_simulations}")

            trajectories = self.simulator(self.n_patients, param_sampler, t_eval)
            return compute_summary_stats(
                trajectories,
                t_eval,
                self.biomarker_names,
                self.std_params,
            )

        if n_jobs is None or n_jobs == 1:
            simulated_stats: List[np.ndarray] = []
            for sim_idx in range(self.n_simulations):
                simulated_stats.append(_one_sim(sim_idx))
        else:
            simulated_stats = Parallel(n_jobs=n_jobs)(
                delayed(_one_sim)(sim_idx)
                for sim_idx in range(self.n_simulations)
            )

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
        n_jobs: int = 1,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Full evaluation: run simulations and compute log-likelihood."""
        mu_hat, sigma_hat, sim_stats = self.run_simulations(
            param_sampler,
            t_eval,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        log_likelihood, details = self.compute_log_likelihood(mu_hat, sigma_hat)
        details['simulated_stats'] = sim_stats
        return log_likelihood, details


class _PhysioRejection(Exception):
    """Internal signal that a simulation failed physiological validity."""

    def __init__(self, valid_fraction: float):
        super().__init__(f"valid_fraction={valid_fraction:.3f}")
        self.valid_fraction = float(valid_fraction)


def _evaluate_system_logsl_core(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    n_jobs: int = 1,
) -> Tuple[float | None, Dict[str, np.ndarray] | None, np.ndarray | None, List[str] | None, List[str] | None]:
    """Core synthetic-likelihood evaluation returning diagnostics alongside score."""
    from llmode import initial_condition_utils
    from llmode import ode_simulator
    from llmode import param_utils

    # Load configuration and observed data (cached per problem).
    obs = get_observed_summary(problem_name)
    config = obs["config"]
    biomarker_names = obs["biomarker_names"]
    all_biomarker_names = obs["all_biomarker_names"]
    t_eval = obs["t_eval"]
    std_params = obs["std_params"]
    s_obs = obs["s_obs"]
    stat_names = obs["stat_names"]

    # If no parameter distributions are available, we cannot form a meaningful
    # synthetic likelihood; treat this as unevaluable.
    if param_distributions is None:
        return None, None, None, None, None

    def param_sampler(n: int) -> np.ndarray:
        return param_utils.sample_params_from_distributions(
            param_distributions,
            n_samples=n,
            distribution="lognormal",
        )

    # Choose simulator backend.
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
                random_seed=config.get('random_seed', None),
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

            # Drop patients whose trajectories are numerically or physiologically invalid.
            valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
                trajectories,
                config=config,
                check_nans=True,
            )
            valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
            if valid_fraction < 0.7:
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
                random_seed=config.get('random_seed', None),
            )
            param_sets = sampler_fn(n_patients)
            trajectories = ode_simulator.simulate_ode_system(
                system_func=system_func,
                initial_conditions=ic_array,
                time_grid=t_grid,
                params=param_sets,
                method='odeint',
            )
            # Drop patients whose trajectories are numerically or physiologically invalid.
            valid_mask, _issues = ode_simulator.check_trajectory_normal_range_validity(
                trajectories,
                config=config,
                check_nans=True,
            )
            valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
            if valid_fraction < 0.7:
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

    try:
        log_likelihood, details = log_sl.evaluate(
            param_sampler,
            t_eval,
            verbose=False,
            n_jobs=n_jobs,
        )
    except _PhysioRejection as e:
        # System-level rejection: at least one SL simulation had < 80% valid patients.
        prefix = f"Sample {sample_order}: " if sample_order is not None else ""
        # print(
        #     f"{prefix}System rejected: only {e.valid_fraction:.3f} of trajectories "
        #     "fall within physiological normal ranges."
        # )
        return None, None, None, None, None
    if not np.isfinite(log_likelihood):
        return None, None, None, None, None

    if verbose:
        # Optional diagnostics: print variance of each summary statistic under
        # the synthetic model so users can inspect which dimensions are highly
        # variable (and therefore down-weighted in the likelihood).
        sigma_hat = details.get('sigma_hat')
        if sigma_hat is not None and sigma_hat.size > 0:
            variances = np.diag(sigma_hat)
            print("\nVariance of summary statistics under model:")
            for name, var in zip(stat_names, variances):
                print(f"  {name:<40} {var:12.6g}")
            print()
    return float(log_likelihood), details, s_obs, stat_names, biomarker_names


def evaluate_system_logsl(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    n_jobs: int = 1,
) -> float | None:
    """Evaluate a system function via log synthetic likelihood for a given problem."""
    score, _details, _s_obs, _names, _bios = _evaluate_system_logsl_core(
        system_func=system_func,
        problem_name=problem_name,
        param_distributions=param_distributions,
        verbose=verbose,
        sample_order=sample_order,
        backend=backend,
        n_jobs=n_jobs,
    )
    return score


def evaluate_system_logsl_with_feedback(
    system_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    problem_name: str,
    param_distributions: Dict[str, Dict[str, float]] | None,
    verbose: bool = False,
    sample_order: int | None = None,
    backend: str = "cpu",
    n_jobs: int = 1,
) -> Tuple[float | None, str | None]:
    """Evaluate system and return (score, feedback) for LLM-guided optimization."""
    score, details, s_obs, stat_names, biomarker_names = _evaluate_system_logsl_core(
        system_func=system_func,
        problem_name=problem_name,
        param_distributions=param_distributions,
        verbose=verbose,
        sample_order=sample_order,
        backend=backend,
        n_jobs=n_jobs,
    )
    if score is None or details is None or s_obs is None or stat_names is None or biomarker_names is None:
        return None, None

    feedback = generate_llm_feedback(
        log_likelihood=score,
        s_obs=s_obs,
        details=details,
        stat_names=stat_names,
        biomarker_names=biomarker_names,
    )
    return score, feedback
