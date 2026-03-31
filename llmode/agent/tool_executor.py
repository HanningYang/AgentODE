"""Helpers for executing tools over longitudinal trajectory data."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import json
import os
import numpy as np

from llmode.ode import initial_condition_utils
from . import ts_features as tsf


# ---------------------------------------------------------------------------
# Tool registry and execution over trajectory arrays
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Any] = {
    "mean":                    tsf.mean,
    "variance":                tsf.variance,
    "skewness":                tsf.skewness,
    "acf_lag1":                tsf.acf_lag1,
    "acf_lag3":                tsf.acf_lag3,
    "acf_1e_timescale":        tsf.acf_1e_timescale,
    "dominant_frequency":      tsf.dominant_frequency,
    "statav":                  tsf.statav,
    "std_first_difference":    tsf.std_first_difference,
    "permutation_entropy_m3":  tsf.permutation_entropy_m3,
    "time_reversal_asymmetry": tsf.time_reversal_asymmetry,
    "ar_coef_1":               tsf.ar_coef_1,
    "ar_coef_2":               tsf.ar_coef_2,
    "turning_point_rate":      tsf.turning_point_rate,
    "spearman_trend":          tsf.spearman_trend,
    "diff_corr":               tsf.diff_corr,
}


def _load_ts_stats(problem_name: str) -> List[Dict[str, Any]]:
    """Load precomputed time-series stats for a problem."""
    stats_path = os.path.join("workspace", problem_name, "stats", "ts_stats.json")
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
    # Index ts_stats.json by (statistic, variable).
    ts_stats = _load_ts_stats(problem_name)
    index: Dict[tuple[str, str], float | None] = {}
    for row in ts_stats:
        stat = str(row.get("statistic"))
        var = str(row.get("variable"))
        index[(stat, var)] = row.get("value")

    def _lookup_observed(tool_name: str, entry: Dict[str, Any]) -> float | None:
        # Map tool name to ts_stats "statistic".
        stat_key = tool_name
        if tool_name == "dominant_frequency":
            stat_key = "dominant_freq"
        elif tool_name == "std_first_difference":
            stat_key = "std_first_diff"
        elif tool_name == "permutation_entropy_m3":
            stat_key = "perm_entropy_m3"
        elif tool_name == "time_reversal_asymmetry":
            stat_key = "time_reversal_asym"

        if tool_name == "diff_corr":
            var_x = str(entry.get("variable_x"))
            var_y = str(entry.get("variable_y"))
            # ts_stats uses "var1__var2" ordering.
            pair_key = f"{var_x.lower()}__{var_y.lower()}"
            return index.get((stat_key, pair_key))

        var = str(entry.get("variable"))
        return index.get((stat_key, var.lower()))

    # Build rows: stat, variable label, observed, synthetic.
    rows: List[tuple[str, str, float | None, float | None]] = []
    for res in tool_results:
        tool_name = str(res.get("tool"))
        if tool_name == "diff_corr":
            label = f"{res.get('variable_x')} vs {res.get('variable_y')}"
        else:
            label = str(res.get("variable"))

        obs_val = _lookup_observed(tool_name, res)
        syn_val = res.get("synthetic")
        rows.append((tool_name, label, obs_val, syn_val))

    # Determine column widths.
    stat_col = "stat"
    var_col = "variable"
    obs_col = "observed"
    syn_col = "synthetic"

    stat_width = max(len(stat_col), *(len(r[0]) for r in rows)) if rows else len(stat_col)
    var_width = max(len(var_col), *(len(r[1]) for r in rows)) if rows else len(var_col)
    obs_width = len(obs_col)
    syn_width = len(syn_col)

    def _fmt_val(v: float | None) -> str:
        if v is None:
            return "NA"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return "NA"

    header = (
        f"{stat_col:<{stat_width}}  "
        f"{var_col:<{var_width}}  "
        f"{obs_col:>{obs_width}}  "
        f"{syn_col:>{syn_width}}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for stat, var, obs_val, syn_val in rows:
        obs_str = _fmt_val(obs_val)
        syn_str = _fmt_val(syn_val)
        lines.append(
            f"{stat:<{stat_width}}  "
            f"{var:<{var_width}}  "
            f"{obs_str:>{obs_width}}  "
            f"{syn_str:>{syn_width}}"
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

    # Aggregate across patients for per-variable feature extraction.
    obs_mean = observed.mean(axis=0)   # shape (T, n_variables)
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

        if tool_name == "diff_corr":
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


__all__ = [
    "TOOL_REGISTRY",
    "format_stat_table",
    "extract_tool_calls",
    "execute_tool_calls",
]
