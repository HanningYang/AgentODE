# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration of a AgentODE experiment."""
from __future__ import annotations
import dataclasses
from typing import Type
import os

from agentode.core import sampler
from agentode.core import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.

    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2
    num_islands: int = 10 
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1 # 0.8 # 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for AgentODE experiments.

    Args:
        experience_buffer: Evolution multi-population settings
        num_samplers (int): Number of parallel samplers
        num_evaluators (int): Number of parallel evaluators
        samples_per_prompt (int): Number of hypotheses per prompt
        evaluate_timeout_seconds (int): Hypothesis evaluation timeout

        structure_backend (str): Backend for ODE structure inference. One of:
            'api'       — direct cloud API via aisuite
            'openwebui' — local LLM served via OpenWebUI
            'local'     — local LLM server running on port 5000 (default)

        api_model (str): Legacy common model name for 'api' and 'openwebui'
            modes. Still used as a fallback when structure_model / param_model
            are not provided.

        structure_model (str | None): Model identifier used for ODE structure
            inference (via Sampler / LocalLLM). If None, falls back to
            api_model.

        param_model (str | None): Model identifier used for parameter
            inference / optimization. If None, falls back to structure_model,
            then api_model.

        openwebui_base_url (str): Base URL for OpenWebUI API.
            Defaults to https://openwebui.uni-freiburg.de/api
            Can be overridden with OPENWEBUI_URL env var.

        use_api (bool): Deprecated — kept only for backwards compatibility
                        in parameter optimisation. New code should prefer
                        `structure_backend` and `param_backend`.

        param_optim_steps (int): Maximum parameter-optimization queries per ODE structure
        param_optim_patience (int): Early stopping patience
        param_optim_rel_improvement (float): Minimum relative logSL improvement to reset patience

        param_optim_extended_steps (int): Maximum parameter-optimization queries per
            ODE structure when the Euclidean (L2) score of this structure beats the
            best score in its island. Must be >= param_optim_steps.

        param_backend (str | None): Backend used specifically for parameter
            inference / optimisation. If None, falls back to structure_backend.

        trajectory_bin_width (float): Default time-bin width (in the same units
            as the `t` column) used for observed vs synthetic trajectory
            comparison figures.
    """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(
        default_factory=ExperienceBufferConfig
    )
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30

    # --- LLM backend ---
    structure_backend: str = os.environ.get(
        "STRUCTURE_BACKEND", os.environ.get("LLM_MODE", "openwebui")
    )  # 'api' | 'openwebui' | 'local'
    param_backend: str | None = os.environ.get(
        "PARAM_BACKEND", os.environ.get("PARAM_LLM_MODE", None)
    )
    api_model: str = os.environ.get("API_MODEL", "openai/gpt-5.2-llmlb")
    structure_model: str | None = os.environ.get("STRUCTURE_MODEL", None)
    param_model: str | None = os.environ.get("PARAM_MODEL", None)
    openwebui_base_url: str = os.environ.get(
        "OPENWEBUI_URL", "https://openwebui.uni-freiburg.de/api"
    )

    # kept for backwards compatibility
    use_api: bool = False
    api_provider: str = "openai"

    # --- Parameter optimisation ---
    param_optim_steps: int = 30
    param_optim_patience: int = 10
    param_optim_rel_improvement: float = 0.1
    param_optim_extended_steps: int = 40

    trajectory_bin_width: float = 14.0


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
