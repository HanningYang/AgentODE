"""Prompt construction utilities for the agent workflow."""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Mapping

import pandas as pd

from .figures import _normalize_columns, _auto_detect_vars


def get_variable_names(problem_name: str) -> List[str]:
    """Return normalised biomarker variable names for a problem."""
    obs_path = os.path.join("data", problem_name, f"{problem_name}.csv")
    df = pd.read_csv(obs_path)
    df_norm = _normalize_columns(df)
    vars_list = _auto_detect_vars(df_norm)
    return list(vars_list)


def load_tool_schemas(problem_name: str) -> list[dict[str, Any]]:
    """Load tool schemas with `{variable_names}` filled for a problem."""
    variable_names = get_variable_names(problem_name)
    placeholder = "{variable_names}"
    joined = ", ".join(variable_names)

    schemas_path = os.path.join(os.path.dirname(__file__), "tool_schemas_stat.json")
    with open(schemas_path, "r", encoding="utf-8") as f:
        schemas: list[dict[str, Any]] = json.load(f)

    def _replace_placeholders(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _replace_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_placeholders(v) for v in obj]
        if isinstance(obj, str):
            return obj.replace(placeholder, joined)
        return obj

    return _replace_placeholders(schemas)


def load_filesystem_schemas(experiment_dir: str) -> list[dict[str, Any]]:
    """Load filesystem tool schemas with ``{experiment_dir}`` filled in.

    Args:
        experiment_dir: Path to the sample directory
            (e.g. ``workspace/aki/logs/run1/sample_0``), injected into any
            ``{experiment_dir}`` placeholders in the schema descriptions.

    Returns:
        List of tool schema dicts ready to pass to the LLM.
    """
    schemas_path = os.path.join(os.path.dirname(__file__), "tool_schemas_filesystem.json")
    with open(schemas_path, "r", encoding="utf-8") as f:
        schemas_str = f.read()

    schemas_str = schemas_str.replace("{experiment_dir}", experiment_dir)
    return json.loads(schemas_str)


def extract_json_from_llm_output(llm_output: str) -> dict:
    """Extract the first top-level JSON object from an LLM output string."""
    start_idx = llm_output.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in LLM output")

    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(llm_output)):
        if llm_output[i] == '{':
            brace_count += 1
        elif llm_output[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        raise ValueError("No complete JSON object found in LLM output")

    try:
        return json.loads(llm_output[start_idx:end_idx])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM output: {e}")


def extract_failure_modes(result_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the `failure_modes` list from a diagnosis JSON object.

    Raises:
        ValueError: If `failure_modes` exists but is not a list.
    """
    failure_modes = result_json.get("failure_modes")
    if failure_modes is None:
        return []
    if not isinstance(failure_modes, list):
        raise ValueError("Expected 'failure_modes' to be a list in diagnosis JSON.")
    return [fm for fm in failure_modes if isinstance(fm, dict)]


def _load_spec_sections(spec_path: str) -> Dict[str, str]:
    """Load a spec file with [SECTION] headers into a dict."""
    sections: Dict[str, List[str]] = {}
    current: str | None = None
    with open(spec_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("[") and line.endswith("]") and len(line) > 2:
                current = line.strip("[]")
                sections.setdefault(current, [])
            else:
                if current is None:
                    # Lines before the first header are ignored.
                    continue
                sections[current].append(raw_line.rstrip("\n"))

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def build_param_prompt(
    spec_path: str,
    prompt_type: str,
    placeholders: Mapping[str, str] | None = None,
    image_placeholders: Mapping[str, str] | None = None,
) -> str:
    """Build a unified parameter prompt from a spec and placeholders.

    The final prompt is composed as:

        spec["ROLE"]
        + "\\n\\n"
        + spec[f"{prompt_type}_TASK"]
        + "\\n\\n"
        + spec["SHARED"]
        + "\\n\\n"
        + spec[prompt_type]

    After assembly, all `placeholders` and `image_placeholders` are applied.
    """
    spec = _load_spec_sections(spec_path)

    role = spec.get("ROLE", "")
    task = spec.get(f"{prompt_type}_TASK", "")
    shared = spec.get("SHARED", "")
    body = spec.get(prompt_type, "")

    prompt_parts = [p for p in (role, task, shared, body) if p]
    prompt = "\n\n".join(prompt_parts)

    # Apply text placeholders (e.g. ode_system, violation_report).
    for key, value in (placeholders or {}).items():
        placeholder = "{" + key + "}"
        prompt = prompt.replace(placeholder, value)

    # Apply image/extra placeholders (e.g. figures_observed, stat_table).
    for key, value in (image_placeholders or {}).items():
        placeholder = "{" + key + "}"
        prompt = prompt.replace(placeholder, value)

    return prompt


def build_figures_block_from_dir(fig_dir: str) -> str:
    """Return a simple newline-separated list of figure paths from a directory."""
    if not os.path.isdir(fig_dir):
        return ""
    files = sorted(
        f for f in os.listdir(fig_dir)
        if f.lower().endswith(".png")
    )
    if not files:
        return ""
    return "\n".join(os.path.join(fig_dir, f) for f in files)


__all__ = [
    "get_variable_names",
    "load_tool_schemas",
    "load_filesystem_schemas",
    "extract_json_from_llm_output",
    "extract_failure_modes",
    "build_param_prompt",
    "build_figures_block_from_dir",
]
