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


def _print_trajectory_value_stats(
    trajectories: np.ndarray,
    biomarker_names: list,
    config: dict,
) -> None:
    """Print actual trajectory value statistics per biomarker.

    For each biomarker reports: value range across all patients/times,
    mean at t=0 and t=end (finite values only), NaN onset info if applicable,
    and how actual range compares to physiological range if configured.
    """
    print("  Trajectory value statistics        :")
    for j, name in enumerate(biomarker_names):
        col = trajectories[:, :, j]  # (n_samples, n_timepoints)
        finite_vals = col[np.isfinite(col)]

        # NaN onset: earliest step at which any NaN appears per patient.
        nan_mask = np.isnan(col)
        nan_count = nan_mask.any(axis=1).sum()
        if nan_count > 0:
            first_nan_steps = np.where(
                nan_mask.any(axis=1),
                np.argmax(nan_mask, axis=1),
                col.shape[1],
            )
            affected_steps = first_nan_steps[first_nan_steps < col.shape[1]]
            nan_note = f"NaN in {nan_count} patients (median onset step {int(np.median(affected_steps))})"
        else:
            nan_note = None

        if finite_vals.size == 0:
            print(f"    {name}: all values are NaN/Inf — no finite values to report")
            continue

        val_min = float(np.min(finite_vals))
        val_max = float(np.max(finite_vals))

        # Mean at first and last time step (finite only).
        t0_finite = col[:, 0][np.isfinite(col[:, 0])]
        t0_str = f"{np.mean(t0_finite):.3g}" if t0_finite.size > 0 else "NaN"
        tend_finite = col[:, -1][np.isfinite(col[:, -1])]
        tend_str = f"{np.mean(tend_finite):.3g}" if tend_finite.size > 0 else "NaN"

        range_str = f"[{val_min:.3g}, {val_max:.3g}]"

        # Compare to physiological range if available.
        phys_range = config.get("biomarkers", {}).get(name, {}).get("physiological_range", {})
        p_min = phys_range.get("normal_min")
        p_max = phys_range.get("normal_max")
        if p_min is not None and p_max is not None:
            phys_str = f" (physiological [{p_min:.3g}, {p_max:.3g}])"
        else:
            phys_str = ""

        line = f"    {name}: range {range_str}{phys_str}, mean t=0→end: {t0_str}→{tend_str}"
        if nan_note:
            line += f"; {nan_note}"
        print(line)


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
    global_allow_negative = bool(config.get("negative", False))

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
            allow_negative = config["biomarkers"][name].get(
                "negative", global_allow_negative
            )
            if allow_negative:
                continue
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

    # Always show what the trajectories actually look like per biomarker.
    _print_trajectory_value_stats(trajectories, biomarker_names, config)

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


def compute_violation_report_dict(
    trajectories: np.ndarray,
    config: dict,
    check_nans: bool = True,
) -> dict:
    """Return a structured violation-report dict for workspace logging.

    The dict mirrors the format injected into constraint-violation prompts::

        {
          "summary": {"n_samples": ..., "n_valid": ..., "n_invalid": ..., "valid_proportion": ...},
          "issue_counts": {
            "<issue_key>": {
              "total": ...,
              "proportion": ...,
              "per_biomarker": {"Creatinine": ..., ...}
            },
            ...
          }
        }

    Only issue types with at least one occurrence are included.
    """
    n_samples, _n_timepoints, _n_biomarkers = trajectories.shape
    biomarker_names = initial_condition_utils.get_biomarker_order(config)
    global_allow_negative = bool(config.get("negative", False))

    valid_mask, issue_counts = check_trajectory_normal_range_validity(
        trajectories, config=config, check_nans=check_nans,
    )
    n_valid = int(valid_mask.sum())
    n_invalid = n_samples - n_valid

    issue_keys = ["has_nan", "has_inf", "negative_values", "outside_normal_range"]
    per_biomarker_counts: Dict[str, Dict[str, int]] = {
        k: {name: 0 for name in biomarker_names} for k in issue_keys
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
            allow_negative = config["biomarkers"][name].get(
                "negative", global_allow_negative
            )
            if allow_negative:
                continue
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

    report_issue_counts: Dict[str, Dict] = {}
    for key in issue_keys:
        total = issue_counts.get(key, 0)
        if total > 0:
            report_issue_counts[key] = {
                "total": int(total),
                "proportion": round(total / n_samples, 4) if n_samples > 0 else 0.0,
                "per_biomarker": {
                    name: per_biomarker_counts[key][name]
                    for name in biomarker_names
                    if per_biomarker_counts[key][name] > 0
                },
            }

    return {
        "summary": {
            "n_samples": n_samples,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "valid_proportion": round(n_valid / n_samples, 4) if n_samples > 0 else 0.0,
        },
        "issue_counts": report_issue_counts,
    }


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
    "compute_violation_report_dict",
    "compute_valid_fraction",
]
