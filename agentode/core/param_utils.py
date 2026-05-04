"""Utilities for parameter distribution inference with LLM."""

import os
import json
import re
from typing import Dict, Any, Optional, Dict as TypingDict

import numpy as np

from agentode.core import code_manipulation


DEBUG_PRINTS = os.environ.get("AGENTODE_DEBUG_PRINTS", "0") == "1"

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
    """Set the global default system docstring used for prompts."""
    global _GLOBAL_DEFAULT_SYSTEM_DOCSTRING
    _GLOBAL_DEFAULT_SYSTEM_DOCSTRING = docstring


def set_default_system_docstring_from_program(
    program: code_manipulation.Program,
    function_name: str = 'system',
) -> None:
    """Derive default system docstring from a specification `Program`."""
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


def count_params_used(code: str, param_array_name: str = 'params') -> int:
    """Return max parameter index used in `code` (+1)."""
    pattern = rf'{re.escape(param_array_name)}\[[^\]]*?(\d+)[^\]]*]'
    matches = re.findall(pattern, code)
    if not matches:
        return 0
    return max(int(idx) for idx in matches) + 1


def validate_param_distributions_format(
    param_distributions: Dict[str, Any],
) -> Dict[str, Any]:
    """Return cleaned param dict with float `mean`/`sd` or raise ValueError.

    Keys are normalised to strings and ordered numerically ("0", "1", ...).
    This guarantees consistent numeric ordering whenever the resulting dict
    is serialized to JSON or logged elsewhere in the codebase.
    """
    if not isinstance(param_distributions, dict):
        raise ValueError("param_distributions must be a JSON object mapping parameter indices to dictionaries.")

    entries: TypingDict[str, Any] = {}
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
        # Preserve optional metadata such as rationale.
        for meta_key, meta_val in value.items():
            if meta_key in ("mean", "sd"):
                continue
            cleaned_entry[meta_key] = meta_val
        entries[str(key)] = cleaned_entry

    if not entries:
        raise ValueError("No valid parameter entries found in param_distributions.")

    # Sort numeric keys first, then any non-numeric keys, for stable output.
    def _sort_key(k: str) -> tuple[int, int | str]:
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, k)

    cleaned: Dict[str, Any] = {}
    for k in sorted(entries.keys(), key=_sort_key):
        cleaned[k] = entries[k]

    return cleaned


def get_default_param_distributions(problem_name: str) -> Dict[str, Any] | None:
    """Return built-in default param priors for `problem_name`."""
    if problem_name == 'aki':
        return {
            "0": {"mean": 0.02, "sd": 0.01, "rationale": "Initial generic prior from the dosc template (parameter 0)."},
            "1": {"mean": 0.015, "sd": 0.0075, "rationale": "Initial generic prior from the dosc template (parameter 1)."},
            "2": {"mean": 0.8, "sd": 0.4, "rationale": "Initial generic prior from the dosc template (parameter 2)."},
        }
    elif problem_name == 'polymer':
        return {
            "0": {
                "mean": 0.01,
                "sd": 0.00000005,
                "rationale": "Initial generic prior from the dosc template (parameter 0).",
            },
            "1": {
                "mean": 0.01,
                "sd": 0.00000005,
                "rationale": "Initial generic prior from the dosc template (parameter 1).",
            },
            "2": {
                "mean": 0.01,
                "sd": 0.00000005,
                "rationale": "Initial generic prior from the dosc template (parameter 2).",
            },
            "3": {
                "mean": 0.01,
                "sd": 0.00000005,
                "rationale": "Initial generic prior from the dosc template (parameter 3).",
            },
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
        param_distributions: JSON-like dict from `prompt_builder.extract_json_from_llm_output`.
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
        samples = rng.normal(
            loc=np.broadcast_to(means, (n_samples, n_params)),
            scale=np.broadcast_to(sds, (n_samples, n_params)),
        )
        return samples

    if distribution == "lognormal":
        # Treat mean/sd as linear-scale moments and convert them to log-space.
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
