"""Population-level variability statistics across patient trajectories."""

import numpy as np


def population_std(
    trajectories: np.ndarray,  # shape (N, T)
) -> float:
    """
    Standard deviation of per-patient means across the population.
    Captures between-patient variability.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    per_patient_means = np.nanmean(trajectories, axis=1)  # shape (N,)
    valid = per_patient_means[~np.isnan(per_patient_means)]
    return float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan


def population_iqr(
    trajectories: np.ndarray,  # shape (N, T)
) -> float:
    """
    Interquartile range of per-patient means across the population.
    Robust measure of between-patient spread.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    per_patient_means = np.nanmean(trajectories, axis=1)  # shape (N,)
    valid = per_patient_means[~np.isnan(per_patient_means)]
    if valid.size == 0:
        return np.nan
    return float(np.percentile(valid, 75) - np.percentile(valid, 25))
