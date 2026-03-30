"""Agent-facing helpers for trajectory plausibility diagnostics.

This module wraps the low-level physiological validity check from
`llmode.ode.ode_simulator` and formats a human-readable report string
that can be injected into LLM prompts.
"""

from typing import Dict, Tuple
import io
import contextlib

import numpy as np

from llmode.ode import initial_condition_utils
from llmode.ode.ode_simulator import check_trajectory_normal_range_validity


def print_unplausible_trajectory_report(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> None:
    """Print a human-readable report of implausible trajectories.

    Args:
        trajectories: (n_samples, n_timepoints, n_biomarkers).
        config: Initial-condition configuration dictionary.
        check_nans: Whether to treat NaNs as implausible.
    """
    n_samples, n_timepoints, n_biomarkers = trajectories.shape
    biomarker_names = initial_condition_utils.get_biomarker_order(config)

    valid_mask, issue_counts = check_trajectory_normal_range_validity(
        trajectories,
        config=config,
        check_nans=check_nans,
    )

    per_biomarker_counts: Dict[str, Dict[str, int]] = {
        "has_nan": {name: 0 for name in biomarker_names},
        "has_inf": {name: 0 for name in biomarker_names},
        "negative_values": {name: 0 for name in biomarker_names},
        "outside_normal_range": {name: 0 for name in biomarker_names},
    }

    for i in range(n_samples):
        traj = trajectories[i]

        if check_nans:
            for j, name in enumerate(biomarker_names):
                if np.any(np.isnan(traj[:, j])):
                    per_biomarker_counts["has_nan"][name] += 1

        for j, name in enumerate(biomarker_names):
            if np.any(np.isinf(traj[:, j])):
                per_biomarker_counts["has_inf"][name] += 1

        for j, name in enumerate(biomarker_names):
            if np.any(traj[:, j] < 0):
                per_biomarker_counts["negative_values"][name] += 1

        for j, name in enumerate(biomarker_names):
            phys_range = config["biomarkers"][name].get("physiological_range")
            if not phys_range:
                continue
            normal_min = phys_range.get("normal_min")
            normal_max = phys_range.get("normal_max")
            if normal_min is None or normal_max is None:
                continue
            vals = traj[:, j]
            if np.any(vals < normal_min) or np.any(vals > normal_max):
                per_biomarker_counts["outside_normal_range"][name] += 1

    n_valid = int(valid_mask.sum())
    n_invalid = n_samples - n_valid

    print("\n[Implausible Trajectories Report]")
    print(f"  Patients (samples)              : {n_samples}")
    print(f"  Time points per trajectory      : {n_timepoints}")
    print(f"  Biomarkers per patient          : {n_biomarkers}")
    print(f"  Valid trajectories              : {n_valid} / {n_samples}")
    print(f"  Invalid trajectories            : {n_invalid}")
    print("  Issue counts across patients    :")

    def _format_per_biomarker(issue_key: str) -> str:
        parts = [
            f"{name} {per_biomarker_counts[issue_key][name]}"
            for name in biomarker_names
            if per_biomarker_counts[issue_key].get(name, 0) > 0
        ]
        return " (" + ", ".join(parts) + ")" if parts else ""

    for key in ["has_nan", "has_inf", "negative_values", "outside_normal_range"]:
        total = issue_counts.get(key, 0)
        suffix = _format_per_biomarker(key)
        print(f"    - {key}: {total}{suffix}")

    if n_invalid == 0:
        print("  All trajectories are within physiological normal ranges.\n")
    else:
        print("============================================\n")


def get_unplausible_trajectory_report(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> str:
    """Capture and return the text from print_unplausible_trajectory_report.

    Used to embed the report in LLM prompts or logs.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_unplausible_trajectory_report(
            trajectories=trajectories,
            config=config,
            check_nans=check_nans,
        )
    return buf.getvalue()


def compute_valid_fraction(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> Tuple[np.ndarray, float]:
    """Return (valid_mask, valid_fraction) for physiological plausibility.

    This wraps `check_trajectory_normal_range_validity` and is intended for
    use in the parameter inference pipeline and diagnostics, keeping metric
    code focused on score computation.
    """
    valid_mask, _issues = check_trajectory_normal_range_validity(
        trajectories,
        config=config,
        check_nans=check_nans,
    )
    valid_fraction = float(valid_mask.sum()) / float(valid_mask.size)
    return valid_mask, valid_fraction


__all__ = [
    "print_unplausible_trajectory_report",
    "get_unplausible_trajectory_report",
    "compute_valid_fraction",
]
