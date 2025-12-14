"""Utilities for parameter distribution inference with LLM."""

import os
import json
import re
from typing import Dict, Any, Optional

import numpy as np

from llmode import code_manipulation


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
        code: Python code string containing parameter references (e.g., params[0], params[5])
        param_array_name: Name of the parameter array (default: 'params')

    Returns:
        Number of parameters used (max_index + 1), or 0 if none found

    Example:
        >>> code = "d_biomarkers[0] = params[0] - params[5] * x"
        >>> count_params_used(code)
        6
    """
    pattern = rf'{param_array_name}\[(\d+)\]'
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
        Formatted prompt with system function inserted before "JSON Output Format"
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

    # Insert function before "JSON Output Format"
    insertion_marker = "JSON Output Format (strictly)"
    if insertion_marker in spec_template:
        parts = spec_template.split(insertion_marker)
        prompt = parts[0] + function_str + "\n\n" + insertion_marker + parts[1]
    else:
        # Fallback: just append at the end
        prompt = spec_template + "\n\n" + function_str

    return prompt


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
        # Defaults derived from the `sample_parameters_set` example, interpreted
        # as lognormal means/SDs on the linear scale.
        return {
            "0": {"mean": 0.03,  "sd": 0.02},
            "1": {"mean": 0.10,  "sd": 0.025},
            "2": {"mean": 0.60,  "sd": 0.30},
            "3": {"mean": 0.04,  "sd": 0.008},
            "4": {"mean": 0.005, "sd": 0.002},
            "5": {"mean": 0.03,  "sd": 0.008},
            "6": {"mean": 0.05,  "sd": 0.015},
        }

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
