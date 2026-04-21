"""Prompt construction utilities for the agent workflow."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Mapping

import pandas as pd


def _resolve_stats_path(problem_name: str) -> str:
    """Return split-aware stats path for prompt metadata.

    Prefers ``ts_stats_train.json`` for split problems and falls back to
    ``ts_stats.json``. As a last resort, accepts the CSV counterparts.
    """
    stats_dir = os.path.join("workspace", problem_name, "stats")
    candidates = [
        os.path.join(stats_dir, "ts_stats_train.json"),
        os.path.join(stats_dir, "ts_stats.json"),
        os.path.join(stats_dir, "ts_stats_train.csv"),
        os.path.join(stats_dir, "ts_stats.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No stats file found in workspace for problem '{problem_name}'. "
        f"Expected one of: ts_stats_train.json, ts_stats.json, "
        f"ts_stats_train.csv, ts_stats.csv."
    )


def get_variable_names(problem_name: str) -> List[str]:
    """Return normalized biomarker variable names from precomputed stats."""
    stats_path = _resolve_stats_path(problem_name)

    if stats_path.endswith(".json"):
        with open(stats_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        variables = sorted(
            {
                str(row.get("variable", "")).strip().lower()
                for row in rows
                if str(row.get("variable", "")).strip()
                and "__" not in str(row.get("variable", ""))
            }
        )
        if variables:
            return variables
    else:
        df = pd.read_csv(stats_path)
        if "variable" in df.columns:
            variables = sorted(
                {
                    str(value).strip().lower()
                    for value in df["variable"].dropna().tolist()
                    if str(value).strip() and "__" not in str(value)
                }
            )
            if variables:
                return variables

    raise ValueError(f"Could not derive variable names from stats file: {stats_path}")


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


def _iter_json_object_candidates(llm_output: str) -> list[str]:
    """Return plausible JSON object substrings from an LLM response.

    The model may wrap JSON in markdown fences or add explanatory text before
    the object. We therefore try a few bounded candidates before giving up.
    """
    candidates: list[str] = []
    text = llm_output.strip()
    if text:
        candidates.append(text)

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", llm_output, flags=re.DOTALL | re.IGNORECASE):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    for match in re.finditer(r"\{", llm_output):
        start_idx = match.start()
        snippet = llm_output[start_idx:].lstrip()
        if snippet:
            candidates.append(snippet)

    return candidates


def _decode_first_json_object(candidate: str) -> dict[str, Any] | None:
    """Decode the first JSON object in a string, if present."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        start_idx = match.start()
        try:
            parsed, _ = decoder.raw_decode(candidate[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_json_from_llm_output(llm_output: str) -> dict:
    """Extract the first top-level JSON object from an LLM output string."""
    for candidate in _iter_json_object_candidates(llm_output):
        parsed = _decode_first_json_object(candidate)
        if parsed is not None:
            return parsed

    preview = llm_output.strip().replace("\n", "\\n")[:200]
    raise ValueError(f"Invalid JSON in LLM output: could not decode any object. Preview: {preview}")


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

    # Prepend turn-budget notice for multi-turn prompts only.
    if prompt_type in ("DIAGNOSIS", "UPDATE"):
        budget_notice = (
            "You are operating under a 5-turn budget. Before every response,\n"
            "check: if this is turn 4 or later, return the output contract\n"
            "JSON immediately regardless of whether tool calls are complete.\n"
            "A partial answer with JSON is always better than no JSON."
        )
        prompt = budget_notice + "\n\n" + prompt

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
