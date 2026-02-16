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

"""Configuration of a LLMODE experiments
."""
from __future__ import annotations

import dataclasses
from typing import Type
import os

from llmode import sampler
from llmode import evaluator


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
    num_islands: int = 5 # 10 
    reset_period: int = 3 * 4 * 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.8 # 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMODE experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
       use_api (bool): API usage flag
       api_model (str): Model name for remote APIs
       api_provider (str): Which API provider to use ('openai' or 'deepseek')
       param_optim_steps (int): Maximum number of parameter-optimization queries per ODE structure
       param_optim_patience (int): Patience for early stopping; stop if relative logSL improvement or null-score streak exceeds this.
       param_optim_rel_improvement (float): Minimum relative improvement in logSL required to reset patience (e.g. 0.1 for 10%).
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30  
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"
    api_provider: str = "openai"
    param_optim_steps: int = 10
    param_optim_patience: int = 3
    param_optim_rel_improvement: float = 0.1


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
