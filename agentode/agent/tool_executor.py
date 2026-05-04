"""Helpers for executing tools over longitudinal trajectory data."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import json
import os
from pathlib import Path
import numpy as np

from agentode.ode import initial_condition_utils
from . import ts_features as tsf

TOOL_REGISTRY: Dict[str, Any] = {
    "mean":               tsf.mean,
    "std":                tsf.std,
    "iqr":                tsf.iqr,
    "skewness":           tsf.skewness,
    "outlier_frac":       tsf.outlier_frac,
    "value_range":        tsf.value_range,
    "acf_lag1":           tsf.acf_lag1,
    "acf_lag2":           tsf.acf_lag2,
    "acf_lag3":           tsf.acf_lag3,
    "acf_first_zero":     tsf.acf_first_zero,
    "dominant_freq":      tsf.dominant_freq,
    "spectral_entropy":   tsf.spectral_entropy,
    "statav":             tsf.statav,
    "std_first_diff":     tsf.std_first_difference,
    "perm_entropy_m3":    tsf.permutation_entropy_m3,
    "lz_complexity":      tsf.lempel_ziv_complexity,
    "dfa_alpha":          tsf.dfa_alpha,
    "spectral_slope":     tsf.spectral_slope,
    "ar_coef_1":          tsf.ar_coef_1,
    "exp_smooth_alpha":   tsf.exp_smoothing_alpha,
    "turning_point_rate": tsf.turning_point_rate,
    "spearman_trend":     tsf.spearman_trend,
    "mean_crossing_rate": tsf.mean_crossing_rate,
    "mean_abs_diff":      tsf.mean_abs_diff,
    "level_corr":         tsf.level_corr,
    "diff_corr":          tsf.diff_corr,
    "ccf_lag1":           tsf.ccf_lag1,
}

# Table formatting precision by statistic.
_DECIMALS: Dict[str, int] = {
    "mean":               3,
    "std":                3,
    "iqr":                3,
    "skewness":           3,
    "outlier_frac":       4,
    "value_range":        3,
    "acf_lag1":           4,
    "acf_lag2":           4,
    "acf_lag3":           4,
    "acf_first_zero":     2,
    "dominant_freq":      6,
    "spectral_entropy":   4,
    "statav":             4,
    "std_first_diff":     3,
    "perm_entropy_m3":    4,
    "lz_complexity":      4,
    "dfa_alpha":          4,
    "spectral_slope":     4,
    "ar_coef_1":          4,
    "exp_smooth_alpha":   4,
    "turning_point_rate": 4,
    "spearman_trend":     4,
    "mean_crossing_rate": 4,
    "mean_abs_diff":      3,
    "level_corr":         4,
    "diff_corr":          4,
    "ccf_lag1":           4,
}


def _load_ts_stats(problem_name: str) -> List[Dict[str, Any]]:
    """Load precomputed time-series stats for a problem.

    Prefers ``ts_stats_train.json`` (train split) when it exists, falling back
    to ``ts_stats.json`` for problems without a train/test split.
    """
    stats_dir = os.path.join("workspace", problem_name, "stats")
    train_path = os.path.join(stats_dir, "ts_stats_train.json")
    default_path = os.path.join(stats_dir, "ts_stats.json")
    stats_path = train_path if os.path.exists(train_path) else default_path
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_stat_table(
    problem_name: str,
    tool_results: List[Dict[str, Any]],
) -> str:
    """Format a comparison table for selected statistics.

    Args:
        problem_name: Name of the problem (e.g. 'aki'), used to locate
            `workspace/{problem_name}/stats/ts_stats.json`.
        tool_results: Output from `execute_tool_calls`.

    Returns:
        A formatted multiline string suitable for inserting into `{stat_table}`.
    """
    # Normalize variable names so lookups stay case-insensitive.
    ts_stats = _load_ts_stats(problem_name)
    index: Dict[tuple[str, str], float | None] = {}
    for row in ts_stats:
        stat = str(row.get("statistic"))
        var = str(row.get("variable")).lower()
        index[(stat, var)] = row.get("value")

    def _lookup_observed(tool_name: str, entry: Dict[str, Any]) -> float | None:
        stat_key = tool_name

        if tool_name in ("diff_corr", "level_corr", "ccf_lag1"):
            var_x = str(entry.get("variable_x"))
            var_y = str(entry.get("variable_y"))
            # ts_stats stores paired variables as "var1__var2".
            pair_key = f"{var_x.lower()}__{var_y.lower()}"
            return index.get((stat_key, pair_key))

        var = str(entry.get("variable"))
        return index.get((stat_key, var.lower()))

    # Keep only the most recent result for repeated tool calls.
    seen: Dict[tuple[str, str], int] = {}
    for i, res in enumerate(tool_results):
        tool_name = str(res.get("tool"))
        if tool_name in ("diff_corr", "level_corr", "ccf_lag1"):
            key = (tool_name, f"{res.get('variable_x')}__{res.get('variable_y')}")
        else:
            key = (tool_name, str(res.get("variable")))
        seen[key] = i
    deduped = [tool_results[i] for i in sorted(seen.values())]

    rows: List[tuple[str, str, float | None, float | None]] = []
    for res in deduped:
        tool_name = str(res.get("tool"))
        if tool_name in ("diff_corr", "level_corr", "ccf_lag1"):
            label = f"{res.get('variable_x')} vs {res.get('variable_y')}"
        else:
            label = str(res.get("variable"))

        obs_val = _lookup_observed(tool_name, res)
        if obs_val is None or (isinstance(obs_val, float) and np.isnan(obs_val)):
            continue
        syn_val = res.get("synthetic")
        rows.append((tool_name, label, obs_val, syn_val))

    stat_col = "stat"
    var_col = "variable"
    obs_col = "observed"
    syn_col = "synthetic"
    gap_col = "gap"
    rel_gap_col = "rel_gap"

    stat_width = max(len(stat_col), *(len(r[0]) for r in rows)) if rows else len(stat_col)
    var_width = max(len(var_col), *(len(r[1]) for r in rows)) if rows else len(var_col)
    obs_width = len(obs_col)
    syn_width = len(syn_col)
    gap_width = max(len(gap_col), 8)
    rel_gap_width = max(len(rel_gap_col), 8)

    def _fmt_val(v: float | None, decimals: int = 4) -> str:
        if v is None:
            return "NA"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return "NA"

    def _fmt_gap(obs: float | None, syn: float | None) -> str:
        if obs is None or syn is None:
            return "NA"
        try:
            gap = float(syn) - float(obs)
            return f"{gap:+.4f}"
        except (TypeError, ValueError):
            return "NA"

    def _fmt_rel_gap(obs: float | None, syn: float | None) -> str:
        if obs is None or syn is None:
            return "NA"
        try:
            o, s = float(obs), float(syn)
            denom = abs(o)
            if denom < 1e-12:
                return "NA"
            return f"{(s - o) / denom * 100:+.1f}%"
        except (TypeError, ValueError):
            return "NA"

    header = (
        f"{stat_col:<{stat_width}}  "
        f"{var_col:<{var_width}}  "
        f"{obs_col:>{obs_width}}  "
        f"{syn_col:>{syn_width}}  "
        f"{gap_col:>{gap_width}}  "
        f"{rel_gap_col:>{rel_gap_width}}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for stat, var, obs_val, syn_val in rows:
        decimals = _DECIMALS.get(stat, 4)
        obs_str = _fmt_val(obs_val, decimals)
        syn_str = _fmt_val(syn_val, decimals)
        gap_str = _fmt_gap(obs_val, syn_val)
        rel_gap_str = _fmt_rel_gap(obs_val, syn_val)
        lines.append(
            f"{stat:<{stat_width}}  "
            f"{var:<{var_width}}  "
            f"{obs_str:>{obs_width}}  "
            f"{syn_str:>{syn_width}}  "
            f"{gap_str:>{gap_width}}  "
            f"{rel_gap_str:>{rel_gap_width}}"
        )

    return "\n".join(lines)


def extract_tool_calls(result_json: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract the `tool_calls` list from a diagnosis JSON object."""
    tool_calls = result_json.get("tool_calls")
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise ValueError("Expected 'tool_calls' to be a list in diagnosis JSON.")
    return [tc for tc in tool_calls if isinstance(tc, dict)]


def _get_axis_index(biomarker_order: Sequence[str], name: str) -> int:
    """Return axis index for a given biomarker name with a clear error."""
    lower_order = [b.lower() for b in biomarker_order]
    try:
        return lower_order.index(name.lower())
    except ValueError as e:
        raise KeyError(
            f"Variable {name!r} not found in biomarker_order {list(biomarker_order)}."
        ) from e


def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    observed: np.ndarray,     # shape (N, T, n_variables)
    synthetic: np.ndarray,    # shape (N, T, n_variables)
    time_index: np.ndarray,   # shape (T,)
    config: dict,
) -> List[Dict[str, Any]]:
    """Execute a batch of tool calls on observed and synthetic trajectories.

    Args:
        tool_calls: Parsed tool call objects from the LLM.
        observed: Array of observed trajectories, shape (N, T, n_variables).
        synthetic: Array of synthetic trajectories, same shape as observed.
        time_index: Time grid of length T.
        config: Initial condition configuration used to derive biomarker order.
    """
    biomarker_order = initial_condition_utils.get_biomarker_order(config)

    # Compute features on the cohort mean trajectory for each variable.
    obs_mean = observed.mean(axis=0)
    syn_mean = synthetic.mean(axis=0)

    results: List[Dict[str, Any]] = []
    for call in tool_calls:
        tool_name = call.get("tool", "")
        arguments = {k.lower(): v for k, v in (call.get("arguments") or {}).items()}
        reason = call.get("reason")

        tool_name = tool_name.split(".")[-1]
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            raise KeyError(f"Unknown tool {tool_name!r} in TOOL_REGISTRY.")

        if tool_name in ("diff_corr", "level_corr", "ccf_lag1"):
            var_x = arguments["variable_x"]
            var_y = arguments["variable_y"]
            idx_x = _get_axis_index(biomarker_order, var_x)
            idx_y = _get_axis_index(biomarker_order, var_y)
            result = {
                "tool":       tool_name,
                "variable_x": var_x,
                "variable_y": var_y,
                "observed":   fn(obs_mean[:, idx_x], obs_mean[:, idx_y]),
                "synthetic":  fn(syn_mean[:, idx_x], syn_mean[:, idx_y]),
                "reason":     reason,
            }

        elif tool_name == "spearman_trend":
            var = arguments["variable"]
            idx = _get_axis_index(biomarker_order, var)
            result = {
                "tool":      tool_name,
                "variable":  var,
                "observed":  fn(obs_mean[:, idx], time_index),
                "synthetic": fn(syn_mean[:, idx], time_index),
                "reason":    reason,
            }

        else:
            var = arguments["variable"]
            idx = _get_axis_index(biomarker_order, var)
            result = {
                "tool":      tool_name,
                "variable":  var,
                "observed":  fn(obs_mean[:, idx]),
                "synthetic": fn(syn_mean[:, idx]),
                "reason":    reason,
            }

        results.append(result)

    return results


_ITERATION_FILE_MAP: Dict[str, str] = {
    "params":           "params.json",
    "log_sl":           "log_sl.json",
    "violation_report": "violation_report.json",
}


def execute_read_iteration(
    iteration: int,
    file: str,
    experiment_dir: str,
) -> dict:
    """Read a specific file from a past iteration directory.

    Args:
        iteration: Integer iteration number (matches ``iter_N`` directory name).
        file: Logical file key — one of ``"params"``, ``"log_sl"``,
            ``"violation_report"``.
        experiment_dir: Path to the sample directory
            (e.g. ``workspace/aki/logs/run1/sample_0``).

    Returns:
        Parsed JSON content of the requested file, or an error dict if the
        file does not exist.
    """
    filename = _ITERATION_FILE_MAP.get(file)
    if filename is None:
        return {
            "error": (
                f"Unknown file key {file!r}. "
                f"Valid keys: {list(_ITERATION_FILE_MAP)}."
            )
        }

    path = Path(experiment_dir) / f"iter_{iteration}" / filename
    if not path.exists():
        return {
            "error": (
                f"{file} is not available for iter_{iteration}. "
                f"Check the iteration type in the index to see "
                f"which files are available for this iteration type."
            )
        }
    return json.loads(path.read_text(encoding="utf-8"))


def execute_filesystem_tool_calls(
    tool_calls: List[Dict[str, Any]],
    experiment_dir: str,
) -> List[Dict[str, Any]]:
    """Execute a batch of filesystem tool calls and return their results.

    Args:
        tool_calls: Parsed tool call objects from the LLM (same format as
            ``extract_tool_calls`` output).
        experiment_dir: Path to the sample directory passed to
            ``execute_read_iteration``.

    Returns:
        List of dicts with ``tool``, ``arguments``, and ``result`` keys.
    """
    results = []
    for call in tool_calls:
        tool_name = str(call.get("tool", "")).split(".")[-1]
        arguments = {k.lower(): v for k, v in (call.get("arguments") or {}).items()}

        if tool_name == "read_iteration":
            result = execute_read_iteration(
                iteration=int(arguments.get("iteration", -1)),
                file=str(arguments.get("file", "")),
                experiment_dir=experiment_dir,
            )
        else:
            result = {"error": f"Unknown filesystem tool {tool_name!r}."}

        results.append({"tool": tool_name, "arguments": arguments, "result": result})

    return results


__all__ = [
    "TOOL_REGISTRY",
    "_DECIMALS",
    "format_stat_table",
    "extract_tool_calls",
    "execute_tool_calls",
    "execute_read_iteration",
    "execute_filesystem_tool_calls",
]
