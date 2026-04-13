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

    # First, clean all entries keyed by their string index.
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
        # Preserve optional rationale or other metadata fields.
        for meta_key, meta_val in value.items():
            if meta_key in ("mean", "sd"):
                continue
            cleaned_entry[meta_key] = meta_val
        entries[str(key)] = cleaned_entry

    if not entries:
        raise ValueError("No valid parameter entries found in param_distributions.")

    # Rebuild in numeric key order (0,1,2,...) with any non-numeric keys,
    # if present, appended afterwards in lexicographic order. This ensures
    # deterministic ordering whenever the dict is printed or saved.
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
        # Defaults for the 3-biomarker AKI system (Creatinine, BUN, Potassium)
        # return {
        #     "0": {"mean": 0.05, "sd": 0.04, "rationale": "Lower minimum renal function to increase baseline Cr/BUN, helping correct Cr quantile location."},
        #     "1": {"mean": 1.1, "sd": 0.3, "rationale": "Slightly higher max renal function with more variability to increase Cr spread and reduce potassium autocorrelation."},
        #     "2": {"mean": 0.003, "sd": 0.002, "rationale": "Reduced recovery rate to weaken systematic downward trends in all biomarkers."},
        #     "3": {"mean": 0.03, "sd": 0.02, "rationale": "Lower Cr production to reduce baseline creatinine toward observed quantiles."},
        #     "4": {"mean": 0.15, "sd": 0.08, "rationale": "Increased renal Cr clearance to lower Cr baseline and increase spread via injury feedback variability."},
        #     "5": {"mean": 0.015, "sd": 0.012, "rationale": "Reduced non-renal Cr clearance mean but kept wide SD to maintain inter-patient variability in baseline."},
        #     "6": {"mean": 0.8, "sd": 0.4, "rationale": "Lower reference Cr with wider SD to accommodate diverse patient baselines and increase Cr spread."},
        #     "7": {"mean": 0.25, "sd": 0.15, "rationale": "Lower BUN production to reduce baseline and weaken downward population trend."},
        #     "8": {"mean": 0.08, "sd": 0.05, "rationale": "Reduced renal BUN clearance to weaken excessive downward trend in BUN and coupled K."},
        #     "9": {"mean": 0.015, "sd": 0.012, "rationale": "Reduced non-renal BUN clearance mean but kept wide SD to maintain variability."},
        #     "10": {"mean": 12.0, "sd": 6.0, "rationale": "Lower BUN reference with wider SD to increase spread and accommodate heterogeneous set-points."},
        #     "11": {"mean": 4.2, "sd": 0.6, "rationale": "Slightly higher K set-point with wider SD to increase potassium variability and reduce autocorrelation."},
        #     "12": {"mean": 0.8, "sd": 0.3, "rationale": "Increased K buffering rate to accelerate potassium fluctuations and reduce lag-1 autocorrelation."},
        #     "13": {"mean": 0.04, "sd": 0.025, "rationale": "Reduced renal K excretion rate to weaken excessive downward population trend in potassium."},
        #     "14": {"mean": 0.04, "sd": 0.02, "rationale": "Increased baseline K release from impaired renal function to counteract excessive downward trend."},
        #     "15": {"mean": 0.003, "sd": 0.0025, "rationale": "Reduced K sensitivity to Cr to moderate injury-driven K release, aiding trend correction."},
        #     "16": {"mean": 0.0006, "sd": 0.0004, "rationale": "Reduced K sensitivity to BUN to further moderate injury-driven K release."},
        #     "17": {"mean": 2.5, "sd": 1.0, "rationale": "Lower Cr severity threshold with wider SD to increase patient heterogeneity in injury response."},
        #     "18": {"mean": 35.0, "sd": 18.0, "rationale": "Lower BUN severity threshold with wider SD to increase heterogeneity in urea-driven K release."},
        #     "19": {"mean": 1.2, "sd": 0.7, "rationale": "Lower Cr injury threshold with wider SD to increase inter-patient variability in injury onset timing."},
        #     "20": {"mean": 25.0, "sd": 15.0, "rationale": "Lower BUN injury threshold with wider SD to increase variability in injury feedback dynamics."},
        #     "21": {"mean": 0.12, "sd": 0.1, "rationale": "Slightly reduced injury feedback strength mean but increased SD to allow both smooth and dynamic trajectories across patients."},
        #     "22": {"mean": 0.3, "sd": 0.25, "rationale": "Reduced Cr saturation coefficient to allow more clearance variability, reducing Cr autocorrelation and increasing spread."},
        #     "23": {"mean": 0.02, "sd": 0.018, "rationale": "Slightly reduced BUN reabsorption to moderate upward pressure on BUN baseline while keeping variability."},
        #     "24": {"mean": 0.001, "sd": 0.0008, "rationale": "Reduced acidosis-driven K shift to further weaken excessive downward potassium trend."},
        # }
        return {
            "0": {"mean": 0.02, "sd": 0.01, "rationale": "Initial generic prior from the dosc template (parameter 0)."},
            "1": {"mean": 0.015, "sd": 0.0075, "rationale": "Initial generic prior from the dosc template (parameter 1)."},
            "2": {"mean": 0.8, "sd": 0.4, "rationale": "Initial generic prior from the dosc template (parameter 2)."},
            "3": {"mean": 0.02, "sd": 0.01, "rationale": "Initial generic prior from the dosc template (parameter 3)."},
            "4": {"mean": 0.015, "sd": 0.0075, "rationale": "Initial generic prior from the dosc template (parameter 4)."},
            "5": {"mean": 0.03, "sd": 0.015, "rationale": "Initial generic prior from the dosc template (parameter 5)."},
        }
    elif problem_name == 'dosc':
        return {
            "0": {
                "mean": 0.5,
                "sd": 0.5,
                "rationale": "Initial generic prior from the dosc template (parameter 0).",
            },
            "1": {
                "mean": 0.5,
                "sd": 0.5,
                "rationale": "Initial generic prior from the dosc template (parameter 1).",
            },
        }
    elif problem_name == 'lvk':
        # return {
        #     "0": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,100] over t=[0,50].",
        #     },
        #     "1": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
        #     },
        # }
        # return {
        #     "0": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,100] over t=[0,50].",
        #     },
        #     "1": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
        #     },
        #     "2": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
        #     },
        #     "3": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
        #     },
        #     "4": {
        #         "mean": 0.02,
        #         "sd": 0.01,
        #         "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
        #     },
        # }
        return {
            "0": {
                "mean": 0.65,
                "sd": 0.18,
                "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,100] over t=[0,50].",
            },
            "1": {
                "mean": 0.045,
                "sd": 0.015,
                "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
            },
            "2": {
                "mean": 0.55,
                "sd": 0.14,
                "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
            },
            "3": {
                "mean": 0.065,
                "sd": 0.02,
                "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
            },
            "4": {
                "mean": 28.0,
                "sd": 6.0,
                "rationale": "Small growth rate prior so template exponential trajectories stay within valid range [0,200] over t=[0,50].",
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
