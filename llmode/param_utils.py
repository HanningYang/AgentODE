"""Utilities for parameter distribution inference with LLM."""

import os
import json
import re
from typing import Dict, Any, Optional, Dict as TypingDict

import numpy as np
import requests
import http.client

from llmode import code_manipulation


# Global flag to control debug printing for LLM prompts/outputs.
DEBUG_PRINTS = os.environ.get("LLMODE_DEBUG_PRINTS", "0") == "1"

# Fallback docstring used only when we cannot extract a canonical system
# docstring from the current problem specification.
DEFAULT_SYSTEM_DOCSTRING = (
    "Symbolic skeleton for an ODE system.\n\n"
    "    Args:\n"
    "        biomarkers: A numpy array of biomarkers.\n"
    "        params: Array of numeric constants or parameters to be optimized.\n\n"
    "    Returns:\n"
    "        A numpy array representing the temporal derivatives (d/dt)\n"
    "        of the biomarkers."
)

_GLOBAL_DEFAULT_SYSTEM_DOCSTRING: Optional[str] = None


def set_default_system_docstring(docstring: str) -> None:
    """Override the global default system docstring used for prompts."""
    global _GLOBAL_DEFAULT_SYSTEM_DOCSTRING
    _GLOBAL_DEFAULT_SYSTEM_DOCSTRING = docstring


def set_default_system_docstring_from_program(
    program: code_manipulation.Program,
    function_name: str = 'system',
) -> None:
    """Set default system docstring from a specification `Program`.

    This should be called once per experiment from the main pipeline after
    parsing the specification file, so that parameter-inference prompts use
    the problem-specific `system` docstring rather than a hard-coded AKI one.
    """
    try:
        func = program.get_function(function_name)
    except ValueError:
        return

    doc = (func.docstring or '').strip()
    if doc:
        set_default_system_docstring(doc)


def _get_default_system_docstring() -> str:
    """Return the active default system docstring."""
    return _GLOBAL_DEFAULT_SYSTEM_DOCSTRING or DEFAULT_SYSTEM_DOCSTRING


def extract_function_from_response(llm_response: str, function_name: str = 'system') -> str:
    """Extract a complete function definition from LLM response with markdown and explanations.

    Args:
        llm_response: Raw LLM response potentially containing markdown, explanations, etc.
        function_name: Name of the function to extract

    Returns:
        Complete function definition as a string

    Raises:
        ValueError: If function cannot be extracted
    """
    # Remove markdown code blocks
    code_block_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        # Use the first code block
        code_candidate = matches[0]
    else:
        # No code block, use the whole response
        code_candidate = llm_response

    # Find the function definition
    lines = code_candidate.split('\n')
    func_start = -1
    func_end = -1

    for i, line in enumerate(lines):
        if line.strip().startswith(f'def {function_name}'):
            func_start = i
            break

    if func_start == -1:
        raise ValueError(f"Could not find function '{function_name}' in response")

    # Find the end of the function (next unindented line or end of code)
    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        # End when we hit an unindented non-empty line that's not a decorator
        if line and not line[0].isspace() and line.strip() and not line.strip().startswith('@'):
            func_end = i
            break

    if func_end == -1:
        func_end = len(lines)

    function_code = '\n'.join(lines[func_start:func_end])

    try:
        program = code_manipulation.text_to_program(function_code)
    except Exception as e:
        raise ValueError(f"Extracted function is not valid Python: {e}") from e

    if not program.functions:
        raise ValueError('No function definitions found in extracted code')

    func = None
    for candidate in program.functions:
        if candidate.name == function_name:
            func = candidate
            break

    if func is None:
        func = program.functions[0]
        func.name = function_name

    else:
        func.name = function_name

    # Clean docstring to keep canonical summary + Args/Returns information.
    doc = func.docstring.strip() if func.docstring else ''
    cleaned_lines = []
    for line in doc.splitlines():
        if 'Improved version' in line:
            continue
        cleaned_lines.append(line)
    cleaned_doc = '\n'.join(line for line in cleaned_lines if line.strip())
    if ('Args:' not in cleaned_doc) or ('Returns:' not in cleaned_doc):
        cleaned_doc = _get_default_system_docstring()
    func.docstring = cleaned_doc

    return str(func)


def count_params_used(code: str, param_array_name: str = 'params') -> int:
    """Count the maximum parameter index used in code.

    Args:
        code: Python code string containing parameter references
              (e.g., params[0], params[5], params[..., 3])
        param_array_name: Name of the parameter array (default: 'params')

    Returns:
        Number of parameters used (max_index + 1), or 0 if none found

    Example:
        >>> code = "d_biomarkers[0] = params[0] - params[5] * x"
        >>> count_params_used(code)
        6
    """
    # Match both simple and batched indexing, e.g.:
    #   params[0]
    #   params[..., 3]
    #   params[:, 2]
    # We only care about the largest integer index that appears inside any
    # bracket that follows `param_array_name`.
    pattern = rf'{re.escape(param_array_name)}\[[^\]]*?(\d+)[^\]]*]'
    matches = re.findall(pattern, code)
    if not matches:
        return 0
    return max(int(idx) for idx in matches) + 1


def build_param_inference_prompt(system_code: str, spec_template_path: str, function_name: str = 'system') -> str:
    """Build a prompt for LLM to infer parameter distributions by inserting system function into spec template.

    Args:
        system_code: The complete system function code
        spec_template_path: Path to the parameter specification template file
        function_name: Name of the system function to analyze

    Returns:
        Formatted prompt with system function inserted before "Instructions"
    """
    # Read the spec template
    with open(spec_template_path, 'r', encoding='utf-8') as f:
        spec_template = f.read()

    # Get the function. When the LLM uses a different function name (e.g. `system_v1`)
    # fall back to the first parsed function and rename it so the downstream prompt
    # always sees a canonical `function_name` definition.
    program = code_manipulation.text_to_program(system_code)
    try:
        func = program.get_function(function_name)
        function_str = str(func)
    except ValueError:
        if not program.functions:
            raise

        # Use the first function as a fallback and rewrite the signature.
        func = program.functions[0]
        function_str = str(func)
        original_header = f'def {func.name}('
        replacement_header = f'def {function_name}('
        if original_header in function_str:
            function_str = function_str.replace(original_header, replacement_header, 1)
        else:
            # If we cannot find the header, just prepend a canonical one.
            function_str = f'def {function_name}{function_str.split("def", 1)[-1]}'

    # Insert function before "Instructions"
    insertion_marker = "Instructions"
    if insertion_marker in spec_template:
        parts = spec_template.split(insertion_marker)
        prompt = parts[0] + function_str + "\n\n" + insertion_marker + parts[1]
    else:
        # Fallback: just append at the end
        prompt = spec_template + "\n\n" + function_str

    return prompt


def _load_opt_sections(problem_name: str) -> dict:
    """Load optimization prompt sections for a given problem.

    Expects a file `specs_parameters/spec_parameters_opt_{problem}.txt` with
    section headers like:

        ### COMMON_INSTRUCTIONS_BEFORE
        ...
        ### MODE_IMPLAUSIBLE
        ...
        ### MODE_LOG_SL
        ...
        ### COMMON_INSTRUCTIONS_AFTER
        ...
    """
    opt_path = os.path.join(
        "specs_parameters", f"spec_parameters_opt_{problem_name}.txt"
    )
    if not os.path.exists(opt_path):
        raise FileNotFoundError(
            f"Optimization spec template not found at {opt_path}"
        )

    with open(opt_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections: dict[str, str] = {}
    current: Optional[str] = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = line[4:].strip()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def _infer_param_names_from_system(
    system_code: str,
    param_array_name: str = "params",
) -> Dict[int, str]:
    """Infer human-readable parameter names from ODE system code.

    Looks for assignments of the form:
        <name> = params[k]
    and returns a mapping {k: name}.
    """
    pattern = rf'\b(\w+)\s*=\s*{param_array_name}\[(\d+)\]'
    matches = re.findall(pattern, system_code)
    mapping: Dict[int, str] = {}
    for var_name, idx_str in matches:
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        mapping[idx] = var_name
    return mapping


def build_param_optimization_prompt(
    system_code: str,
    problem_name: str,
    current_params: Dict[str, Any],
    mode: str,
    implaus_report: Optional[str],
    sl_feedback: Optional[str],
) -> str:
    """Build a prompt for iterative parameter optimization.

    The prompt is constructed as:
      COMMON_INSTRUCTIONS_BEFORE

      (system_code is inserted under the "ODE System" heading)
      (parameter table is inserted under the existing
       "Current parameter distributions (lognormal, values in linear space)" heading)

      MODE_IMPLAUSIBLE + implausibility report  (if mode == 'implausible')
      or
      MODE_LOG_SL + SL feedback                 (if mode == 'log_sl')

      COMMON_INSTRUCTIONS_AFTER
    """
    sections = _load_opt_sections(problem_name)
    before = sections.get("COMMON_INSTRUCTIONS_BEFORE", "")
    after = sections.get("COMMON_INSTRUCTIONS_AFTER", "")
    if mode == "implausible":
        middle = sections.get("MODE_IMPLAUSIBLE", "")
    else:
        middle = sections.get("MODE_LOG_SL", "")

    # Build parameter table without the name column (some parameters may not
    # have clear names); include rationale as a final column.
    indices = _ordered_param_indices(current_params)
    lines = ["idx\tmean(linear)\tsd(linear)\trationale"]
    for idx in indices:
        entry = current_params.get(str(idx), {})
        mean = entry.get("mean", None)
        sd = entry.get("sd", None)
        if mean is None or sd is None:
            continue
        rationale = entry.get("rationale", "")
        # Replace internal newlines/tabs in rationale to keep the table tidy.
        rationale_clean = str(rationale).replace("\n", " ").replace("\t", " ")
        lines.append(f"{idx}\t{float(mean):.4g}\t{float(sd):.4g}\t{rationale_clean}")
    param_block = "\n".join(lines)

    # 1) Insert system_code under the "ODE System" line inside COMMON_INSTRUCTIONS_BEFORE.
    prompt_before = before or ""
    sys_code = system_code.strip()
    if sys_code:
        ode_marker = "ODE System"
        if ode_marker in prompt_before:
            prompt_before = prompt_before.replace(
                ode_marker,
                ode_marker + "\n\n" + sys_code,
                1,
            )
        else:
            # Fallback: prepend the system definition if marker is missing.
            prompt_before = sys_code + "\n\n" + prompt_before

    # 2) Insert parameter table under the existing "Current parameter distributions" heading.
    param_marker = "Current parameter distributions (lognormal, values in linear space)"
    if param_marker in prompt_before:
        prompt_before = prompt_before.replace(
            param_marker,
            param_marker + "\n\n" + param_block,
            1,
        )
    else:
        prompt_before = (
            prompt_before
            + "\n\n"
            + param_marker
            + "\n\n"
            + param_block
        )

    parts: list[str] = []
    if prompt_before.strip():
        parts.append(prompt_before.strip())

    if mode == "implausible":
        if middle.strip():
            parts.append(middle)
            parts.append("")
        if implaus_report:
            # Template already contains the heading for the implausibility
            # report; just append the report text.
            parts.append(implaus_report.strip())
            parts.append("")
    else:
        if middle.strip():
            parts.append(middle)
            parts.append("")
        if sl_feedback:
            # Template already contains the heading "Current best result and
            # failure modes"; just append the feedback text.
            parts.append(sl_feedback.strip())
            parts.append("")

    if after.strip():
        parts.append(after)

    return "\n".join(p for p in parts if p is not None)


def extract_json_from_llm_output(llm_output: str) -> dict:
    """Extract JSON object from LLM output text.

    Args:
        llm_output: Raw output from LLM, possibly containing explanations and JSON

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found in output
    """
    # Try to find JSON in the output
    # Look for content between { and }
    start_idx = llm_output.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in LLM output")

    # Find the matching closing brace
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

    json_str = llm_output[start_idx:end_idx]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM output: {e}")


def _get_api_endpoint_and_headers(config: Any) -> tuple[str, str, dict]:
    """Return (host, path, headers) for the configured API provider."""
    provider = getattr(config, "api_provider", "openai")

    if str(provider).lower() == "deepseek":
        host = "api.deepseek.com"
        path = "/chat/completions"
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
    else:
        host = "api.openai.com"
        path = "/v1/chat/completions"
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")

    if api_key is None:
        env_name = "DEEPSEEK_API_KEY" if str(provider).lower() == "deepseek" else "OPENAI_API_KEY"
        raise RuntimeError(
            "No API key found for provider "
            f"'{provider}'. Please set '{env_name}' or a generic 'API_KEY' environment variable."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LLMODE/1.0",
        "Content-Type": "application/json",
    }
    return host, path, headers


def run_param_optimization_llm(prompt: str, config: Any) -> Optional[Dict[str, Any]]:
    """Query the parameter LLM for an updated set of param distributions.

    This mirrors the behavior of `LocalLLM._do_param_request` / `_do_param_request_api`
    but is invoked from the evaluator during iterative optimization.
    """
    prompt = prompt.strip()
    if not prompt:
        return None

    llm_output: Optional[str] = None

    if not getattr(config, "use_api", False):
        # Local parameter LLM server (port 5001).
        data = {
            "prompt": prompt,
            "repeat_prompt": 1,
            "params": {
                "max_new_tokens": 3072,
                "do_sample": True,
                "temperature": None,
                "top_k": None,
                "top_p": None,
                "add_special_tokens": False,
                "skip_special_tokens": True,
            },
        }
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(
                "http://127.0.0.1:5001/completions",
                data=json.dumps(data),
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json().get("content")
            llm_output = content if isinstance(content, str) else (content[0] if content else None)
        except Exception as e:
            print(f"[Param optimization] Local parameter LLM request failed: {e}")
            return None
    else:
        # API-based parameter inference.
        try:
            host, path, headers = _get_api_endpoint_and_headers(config)
            conn = http.client.HTTPSConnection(host)
            payload = json.dumps(
                {
                    "max_tokens": 8192,
                    "model": getattr(config, "api_model", "gpt-3.5-turbo"),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                }
            )
            conn.request("POST", path, payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            llm_output = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Param optimization] API parameter LLM request failed: {e}")
            return None

    if not llm_output:
        return None

    # Debug: show raw parameter LLM response before JSON extraction.
    if DEBUG_PRINTS:
        print("\n[Raw LLM Response for Parameters]")
        print(llm_output)

    try:
        raw_json = extract_json_from_llm_output(llm_output)
        return validate_param_distributions_format(raw_json)
    except Exception as e:
        print(f"[Param optimization] Failed to parse parameter JSON: {e}")
        return None


def add_params_to_sample_log(sample_log_path: str, param_distributions: dict):
    """Add parameter distributions to an existing sample log JSON file.

    Args:
        sample_log_path: Path to the existing sample JSON file (e.g., logs/test_run1/samples/samples_4.json)
        param_distributions: Dictionary containing parameter distribution info

    Returns:
        Updated data dictionary
    """
    # Read existing log
    with open(sample_log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add parameter distributions
    data['param_distributions'] = param_distributions

    # Write back
    with open(sample_log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return data


def validate_param_distributions_format(
    param_distributions: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that all parameter entries have numeric `mean` and `sd`.

    Rationale or other metadata fields are optional and are preserved when
    present, but every parameter key must at least define `mean` and `sd`
    that can be converted to floats. If this is not the case, a ValueError
    is raised so callers can fall back to problem-level defaults.
    """
    if not isinstance(param_distributions, dict):
        raise ValueError("param_distributions must be a JSON object mapping parameter indices to dictionaries.")

    cleaned: Dict[str, Any] = {}
    for key, value in param_distributions.items():
        if not isinstance(value, dict):
            raise ValueError(f"Parameter entry for key {key!r} must be an object.")
        if "mean" not in value or "sd" not in value:
            raise ValueError(f"Parameter entry for key {key!r} is missing 'mean' or 'sd'.")
        try:
            mean = float(value["mean"])
            sd = float(value["sd"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameter entry for key {key!r} has non-numeric 'mean' or 'sd'.") from e

        cleaned_entry: Dict[str, Any] = {"mean": mean, "sd": sd}
        # Preserve optional rationale or other metadata fields.
        for meta_key, meta_val in value.items():
            if meta_key in ("mean", "sd"):
                continue
            cleaned_entry[meta_key] = meta_val
        cleaned[str(key)] = cleaned_entry

    if not cleaned:
        raise ValueError("No valid parameter entries found in param_distributions.")

    return cleaned


def get_default_param_distributions(problem_name: str) -> Dict[str, Any] | None:
    """Return hard-coded default parameter distributions for a given problem.

    These are used as fallbacks for centralized evaluation (e.g. log synthetic
    likelihood) when no LLM-inferred parameter priors are available yet. The
    structure matches the JSON inferred from LLM output:

        {
          "0": {"mean": ..., "sd": ...},
          "1": {"mean": ..., "sd": ...},
          ...
        }
    """
    if problem_name == 'aki':
        # Defaults for the 3-biomarker AKI system (Creatinine, BUN, Potassium)
        # with a latent recovery process R. These are interpreted as linear-space
        # means/SDs for lognormal sampling downstream.
        # Original defaults (commented out for reference):
        # return {
        #     "0": {"mean": 0.25, "sd": 0.18, "rationale": "Initial effective clearance in hospitalized AKI is often markedly reduced but variable, so R_min is centered low with broad spread while remaining physiologically plausible."},
        #     "1": {"mean": 0.80, "sd": 0.20, "rationale": "Most patients partially recover toward near-normal clearance within a week but not always fully, so R_max is centered below 1 with moderate variability."},
        #     "2": {"mean": 0.015, "sd": 0.010, "rationale": "A recovery rate of ~0.015 1/h implies a 2–4 day timescale consistent with observed 7-day improvement while allowing slower/faster recoveries."},
        #     "3": {"mean": 0.008, "sd": 0.006, "rationale": "Net creatinine input is kept small so Cr dynamics are primarily clearance-driven, but variability allows catabolic states or muscle mass effects to shift trajectories."},
        #     "4": {"mean": 0.050, "sd": 0.025, "rationale": "Renal Cr clearance coefficient is set so typical effective decay rates (k_CrR·R) yield ~1–3 day relaxation toward Cr_ref over 168 hours."},
        #     "5": {"mean": 0.004, "sd": 0.003, "rationale": "Nonrenal Cr clearance is small relative to renal clearance but nonzero to prevent unrealistically persistent elevations when R is very low."},
        #     "6": {"mean": 1.00, "sd": 0.35, "rationale": "Cr_ref represents an individual's baseline setpoint (often ~0.7–1.3 mg/dL) with moderate spread for age/sex/muscle mass differences."},
        #     "7": {"mean": 0.060, "sd": 0.040, "rationale": "BUN production is allowed to be meaningfully positive (catabolism/protein load) with wide variability to permit slow or absent BUN decline despite improving clearance."},
        #     "8": {"mean": 0.008, "sd": 0.004, "rationale": "Renal BUN clearance is set lower than Cr to produce slower, more gradual BUN recovery over days rather than rapid normalization."},
        #     "9": {"mean": 0.0015, "sd": 0.0010, "rationale": "Nonrenal BUN loss is small (e.g., GI losses/dialysis-unmodeled effects) but included to avoid implausibly static BUN in low-R scenarios."},
        #     "10": {"mean": 16.0, "sd": 6.0, "rationale": "BUN_ref is centered near normal range with enough spread to capture baseline differences from diet, liver function, and chronic comorbidity."},
        #     "11": {"mean": 4.10, "sd": 0.25, "rationale": "K_ref is tightly centered around physiologic setpoint to reflect homeostatic control while permitting patient-level variation (meds, acid-base status)."},
        #     "12": {"mean": 0.35, "sd": 0.20, "rationale": "Extrarenal buffering is fast (hours; ~0.35 1/h) to keep potassium nearly flat despite perturbations, with variability for insulin/catecholamine effects."},
        #     "13": {"mean": 0.060, "sd": 0.035, "rationale": "Renal K excretion gain is moderate so that improving R adds stabilization over 1–2 days without dominating the rapid buffering term."},
        #     "14": {"mean": 0.010, "sd": 0.010, "rationale": "Baseline retention/shift source is small (mmol/L/h) so potassium does not drift excessively, but can rise modestly when R is low."},
        #     "15": {"mean": 0.0020, "sd": 0.0020, "rationale": "Creatinine-coupled severity on K is kept weak so even large Cr elevations only add modest K load, yet variability allows occasional clinically relevant hyperkalemic tendency."},
        #     "16": {"mean": 0.00030, "sd": 0.00030, "rationale": "BUN-coupled severity on K is smaller per unit than Cr to prevent unrealistically large K forcing from high BUN while allowing uremic severity to contribute."},
        #     "17": {"mean": 2.0, "sd": 0.8, "rationale": "Creatinine severity threshold is near the cohort's early values so only above-moderate AKI amplifies K shifts, with spread for baseline CKD vs de novo AKI."},
        #     "18": {"mean": 30.0, "sd": 10.0, "rationale": "BUN severity threshold is set around mild elevation so K coupling activates primarily in more uremic patients, with wide spread for nutritional/catabolic differences."},
        # }
        return {
            "0": {"mean": 0.15, "sd": 0.15, "rationale": "Increased SD to allow wider range of initial renal impairment severity."},
            "1": {"mean": 1.05, "sd": 0.3, "rationale": "Increased SD to accommodate greater variability in maximal recovery capacity."},
            "2": {"mean": 0.005, "sd": 0.006, "rationale": "Slightly lower mean recovery rate to slow BUN decline trend; increased SD for diversity."},
            "3": {"mean": 0.07, "sd": 0.05, "rationale": "Increased SD for more inter-patient variability in Cr production."},
            "4": {"mean": 0.1, "sd": 0.08, "rationale": "Increased SD to vary renal Cr clearance capacity across patients."},
            "5": {"mean": 0.015, "sd": 0.015, "rationale": "Increased SD for variability in non-renal Cr clearance."},
            "6": {"mean": 0.5, "sd": 0.3, "rationale": "Increased SD to vary clearance threshold across patients."},
            "7": {"mean": 0.6, "sd": 0.4, "rationale": "Increased mean and SD to raise BUN levels and widen spread, countering over-rapid decline."},
            "8": {"mean": 0.03, "sd": 0.04, "rationale": "Lower mean renal BUN clearance to reduce downward trend; increased SD for variability."},
            "9": {"mean": 0.008, "sd": 0.008, "rationale": "Increased SD for non-renal BUN clearance variability."},
            "10": {"mean": 4.0, "sd": 2.5, "rationale": "Increased SD to vary BUN reference level across patients."},
            "11": {"mean": 4.0, "sd": 0.5, "rationale": "Increased SD for inter-patient variability in K+ homeostatic set-point."},
            "12": {"mean": 0.12, "sd": 0.08, "rationale": "Increased mean buffering rate to reduce K+ autocorrelation (more fluctuations)."},
            "13": {"mean": 0.15, "sd": 0.18, "rationale": "Lower mean renal K+ excretion to reduce over-regulation; increased SD for variability."},
            "14": {"mean": 0.04, "sd": 0.03, "rationale": "Increased baseline K+ release from impaired renal function to raise K+ variability."},
            "15": {"mean": 0.006, "sd": 0.005, "rationale": "Increased SD for variable Cr-driven K+ release strength."},
            "16": {"mean": 0.0015, "sd": 0.0012, "rationale": "Increased SD for variable BUN-driven K+ release."},
            "17": {"mean": 4.0, "sd": 2.0, "rationale": "Increased SD for patient-specific Cr severity triggers."},
            "18": {"mean": 40.0, "sd": 20.0, "rationale": "Increased SD for variable BUN severity thresholds."},
            "19": {"mean": 2.0, "sd": 2.0, "rationale": "Increased SD to vary Cr inhibition sensitivity on recovery."},
            "20": {"mean": 3.0, "sd": 1.5, "rationale": "Increased SD for variable Cr inhibition threshold."},
            "21": {"mean": 0.1, "sd": 0.1, "rationale": "Increased SD to vary BUN inhibition steepness on recovery."},
            "22": {"mean": 50.0, "sd": 25.0, "rationale": "Increased SD for variable BUN inhibition threshold."},
            "23": {"mean": 5.0, "sd": 4.0, "rationale": "Increased SD to vary Cr clearance saturation nonlinearity."},
            "24": {"mean": 5.5, "sd": 1.0, "rationale": "Increased SD for variable hyperkalemia catabolism threshold."},
            "25": {"mean": 0.05, "sd": 0.04, "rationale": "Increased SD for variable hyperkalemia-urea coupling."},
            "26": {"mean": 1.5, "sd": 1.0, "rationale": "Increased SD to vary K+ excretion saturation."},
            "27": {"mean": 100.0, "sd": 80.0, "rationale": "Increased SD for variable recovery saturation timing."},
            "28": {"mean": 3.0, "sd": 2.0, "rationale": "Increased SD for variable Cr threshold for urea boost."},
            "29": {"mean": 0.1, "sd": 0.08, "rationale": "Increased SD for variable Cr-urea coupling."},
            "30": {"mean": 2.0, "sd": 2.5, "rationale": "Lower mean K+ inhibition steepness on BUN clearance to reduce over-inhibition; increased SD."},
            "31": {"mean": 5.2, "sd": 0.8, "rationale": "Increased SD for variable K+ inhibition threshold on BUN clearance."},
            "32": {"mean": 0.05, "sd": 0.04, "rationale": "Increased SD for variable BUN impairment on K+ buffering."},
            "33": {"mean": 60.0, "sd": 30.0, "rationale": "Increased SD for variable BUN impairment threshold."},
            "34": {"mean": 5.0, "sd": 1.0, "rationale": "Increased SD for variable K+ threshold for muscle catabolism."},
            "35": {"mean": 0.02, "sd": 0.02, "rationale": "Increased SD for variable hyperkalemia-Cr production coupling."},
            "36": {"mean": 4.0, "sd": 2.0, "rationale": "Increased SD for variable Cr threshold for acidosis-driven K+ shift."},
            "37": {"mean": 0.01, "sd": 0.01, "rationale": "Increased SD for variable acidosis-K+ coupling."},
        }

        # Second option for 3 variables scenario - Defaults for the 5-biomarker
        # AKI system (Creatinine, BUN, Potassium, Sodium, Hemoglobin) with a
        # latent injury process.
        #
        # return {
        #     "0": {
        #         "mean": 0.03,
        #         "sd": 0.015,
        #         "rationale": "Represents hourly creatinine generation from muscle metabolism, scaled for adult body mass.",
        #     },
        #     "1": {
        #         "mean": 0.06,
        #         "sd": 0.03,
        #         "rationale": "Baseline reciprocal of creatinine's elimination time constant (approx. 16.7 hours for normal GFR).",
        #     },
        #     "2": {
        #         "mean": 0.8,
        #         "sd": 0.4,
        #         "rationale": "Hourly urea nitrogen generation from protein catabolism, scaled for typical AKI catabolic state.",
        #     },
        #     "3": {
        #         "mean": 0.07,
        #         "sd": 0.035,
        #         "rationale": "Baseline reciprocal of BUN's elimination time constant, reflecting renal clearance.",
        #     },
        #     "4": {
        #         "mean": 0.025,
        #         "sd": 0.01,
        #         "rationale": "Hourly net potassium influx from diet and cellular shifts in AKI.",
        #     },
        #     "5": {
        #         "mean": 0.05,
        #         "sd": 0.025,
        #         "rationale": "Baseline reciprocal of potassium's renal excretion time constant.",
        #     },
        #     "6": {
        #         "mean": 0.008,
        #         "sd": 0.004,
        #         "rationale": "Injury growth rate yielding sub-maximal injury over 3-5 days on an hourly scale.",
        #     },
        #     "7": {
        #         "mean": 1.2,
        #         "sd": 0.5,
        #         "rationale": "Maximum injury burden, an order-one scaling factor modulating clearance loss.",
        #     },
        #     "8": {
        #         "mean": 0.5,
        #         "sd": 0.2,
        #         "rationale": "Sensitivity of clearance loss to injury, allowing for partial (not total) GFR loss.",
        #     },
        #     "9": {
        #         "mean": 0.002,
        #         "sd": 0.001,
        #         "rationale": "Coupling coefficient for sodium's influence on potassium, kept small to prevent runaway feedback.",
        #     },
        #     "10": {
        #         "mean": 138.0,
        #         "sd": 2.5,
        #         "rationale": "Sodium homeostatic set point within normal physiological range.",
        #     },
        #     "11": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Reciprocal of sodium correction time constant (approx. 2-3 days) for dilutional changes.",
        #     },
        #     "12": {
        #         "mean": 0.015,
        #         "sd": 0.0075,
        #         "rationale": "Reciprocal of hemoglobin recovery time constant (approx. 3-4 days) accounting for hemodilution and EPO response.",
        #     },
        #     "13": {
        #         "mean": 12.5,
        #         "sd": 1.5,
        #         "rationale": "Hemoglobin homeostatic set point for a hospitalized adult with potential hemodilution.",
        #     },
        # }

        # Legacy defaults for earlier 3-biomarker AKI systems (Creatinine,
        # BUN, Potassium) without explicit Sodium/Hemoglobin states. Kept
        # for reference only; not used in the current 5-variable setup.
        #
        # legacy_defaults_aki_3var = {
        #     "0": {
        #         "mean": 0.01,
        #         "sd": 0.006,
        #         "rationale": "Creatinine production is relatively constant at ~0.01 mg/dL/h in adults, reflecting steady muscle metabolism, with moderate physiological variability.",
        #     },
        #     "1": {
        #         "mean": 0.02,
        #         "sd": 0.012,
        #         "rationale": "Creatinine clearance is reduced in AKI; typical baseline clearance is ~0.02 mL/min/kg, converted to proportional scaling for 7-day dynamics with plausible variation.",
        #     },
        #     "2": {
        #         "mean": 0.5,
        #         "sd": 0.15,
        #         "rationale": "BUN production is approximately 0.5 mg/dL/h from dietary protein metabolism, consistent with normal to mildly elevated rates in hospitalized adults.",
        #     },
        #     "3": {
        #         "mean": 0.04,
        #         "sd": 0.016,
        #         "rationale": "BUN clearance scales with renal function; typical value reflects reduced but non-zero excretion in AKI, with variability due to patient-specific factors.",
        #     },
        #     "4": {
        #         "mean": 0.3,
        #         "sd": 0.12,
        #         "rationale": "Potassium influx from diet and cellular turnover is ~0.3 mmol/L/h, accounting for steady input with expected physiological variation in hospitalized patients.",
        #     },
        #     "5": {
        #         "mean": 0.06,
        #         "sd": 0.018,
        #         "rationale": "Potassium excretion is impaired in AKI; the baseline clearance coefficient is ~0.06 L/h, consistent with reduced renal handling and moderate uncertainty.",
        #     },
        #     "6": {
        #         "mean": 0.01,
        #         "sd": 0.006,
        #         "rationale": "Creatinine's contribution to impairment feedback is moderate; a linear coefficient of ~0.01 reflects its role in signaling renal dysfunction with plausible physiological weight.",
        #     },
        #     "7": {
        #         "mean": 1.0,
        #         "sd": 0.2,
        #         "rationale": "Baseline creatinine level at which impairment feedback is neutral is set at 1.0 mg/dL, representing normal baseline in adults aged 50–64.",
        #     },
        #     "8": {
        #         "mean": 0.02,
        #         "sd": 0.012,
        #         "rationale": "BUN's contribution to impairment feedback is stronger than creatinine due to urea's high concentration; coefficient ~0.02 captures its greater pathophysiological impact.",
        #     },
        #     "9": {
        #         "mean": 15.0,
        #         "sd": 3.0,
        #         "rationale": "Baseline BUN at which feedback is neutral is ~15 mg/dL, reflecting mild-to-moderate accumulation in early AKI without severe uremia.",
        #     },
        #     "10": {
        #         "mean": 0.05,
        #         "sd": 0.03,
        #         "rationale": "Potassium's contribution to impairment feedback is significant due to its role in cellular homeostasis and toxicity; coefficient ~0.05 captures its sensitivity.",
        #     },
        #     "11": {
        #         "mean": 4.5,
        #         "sd": 0.6,
        #         "rationale": "Baseline potassium level for neutral feedback is set at 4.5 mmol/L, representing normal homeostasis in adults and accounting for early AKI-related shifts.",
        #     },
        # }

    return None


def _ordered_param_indices(param_distributions: Dict[str, Any]) -> list[int]:
    """Return parameter indices as a sorted list of ints."""
    indices = []
    for key in param_distributions.keys():
        try:
            indices.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(indices)


def param_distributions_to_arrays(
    param_distributions: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert JSON-style parameter distributions to mean/std arrays.

    Args:
        param_distributions: Mapping like
            {
              "0": {"mean": m0, "sd": s0, ...},
              "1": {"mean": m1, "sd": s1, ...},
              ...
            }

    Returns:
        Tuple of (means, sds), each 1D np.ndarray of shape (n_params,).
        Parameters are ordered by integer index.
    """
    indices = _ordered_param_indices(param_distributions)
    means = []
    sds = []
    for idx in indices:
        entry = param_distributions.get(str(idx), {})
        means.append(float(entry["mean"]))
        sds.append(float(entry["sd"]))
    return np.asarray(means, dtype=float), np.asarray(sds, dtype=float)


def format_param_distributions(
    param_distributions: Dict[str, Any],
) -> str:
    """Return a human-readable table of parameter priors.

    Format:
        idx    name    mean(linear)    sd(linear)

    The `name` column is taken from `entry.get("name")` when present; otherwise
    it falls back to the stringified index.
    """
    indices = _ordered_param_indices(param_distributions)
    lines = ["idx\tname\tmean(linear)\tsd(linear)"]
    for idx in indices:
        entry = param_distributions.get(str(idx), {})
        name = entry.get("name", str(idx))
        mean = entry.get("mean", None)
        sd = entry.get("sd", None)
        if mean is None or sd is None:
            continue
        lines.append(f"{idx}\t{name}\t{float(mean):.4g}\t{float(sd):.4g}")
    return "\n".join(lines)


def sample_params_from_distributions(
    param_distributions: Dict[str, Any],
    n_samples: int,
    random_seed: Optional[int] = None,
    distribution: str = "lognormal",
    shrink: float = 0.9,
) -> np.ndarray:
    """Draw parameter vectors from LLM-inferred marginal distributions.

    Args:
        param_distributions: JSON-like dict from `extract_json_from_llm_output`.
        n_samples: Number of parameter vectors to sample.
        random_seed: Optional RNG seed for reproducibility.
        distribution: One of {"normal", "lognormal"} describing how to sample
            from the provided mean/SD pairs. Defaults to "lognormal" so that
            parameters remain positive while roughly matching the specified
            mean/SD on the original scale.
        shrink: Variance shrinkage factor applied when `distribution="lognormal"`.
            This reduces the long right tail and better mimics a clipped normal.

    Returns:
        Array of shape (n_samples, n_params) with sampled parameter vectors.
        Each parameter `k` is drawn independently according to the chosen
        distribution.
    """
    means, sds = param_distributions_to_arrays(param_distributions)

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    n_params = means.shape[0]
    if distribution == "normal":
        # Broadcast means/sds to (n_samples, n_params) and sample.
        samples = rng.normal(
            loc=np.broadcast_to(means, (n_samples, n_params)),
            scale=np.broadcast_to(sds, (n_samples, n_params)),
        )
        return samples

    if distribution == "lognormal":
        # Interpret `means`/`sds` as the mean and std on the original (linear)
        # scale, and convert to log-space parameters (mu, sigma). Apply an
        # optional shrink factor to sigma to reduce extremely long tails.
        # Assumes strictly positive means.
        eps = 1e-12
        safe_means = np.maximum(means, eps)
        sigma2 = np.log(1.0 + (sds / safe_means) ** 2)
        sigma = np.sqrt(sigma2) * float(shrink)
        mu = np.log(safe_means) - 0.5 * sigma**2

        samples = rng.lognormal(
            mean=mu,
            sigma=sigma,
            size=(n_samples, n_params),
        )
        return samples

    raise ValueError(f"Unsupported distribution type: {distribution!r}")
