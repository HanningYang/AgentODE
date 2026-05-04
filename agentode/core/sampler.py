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

"""Sampling of new program skeletons."""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time
import os
import json
import requests

from agentode.core import evaluator
from agentode.core import buffer
from agentode import config as config_lib

# pip install aisuite openai
import aisuite as ai
from openai import OpenAI as _OpenAIClient


# Global flag to control debug printing for prompts / LLM outputs.
DEBUG_PRINTS = os.environ.get("AGENTODE_DEBUG_PRINTS", "0") == "1"


# aisuite is used for direct cloud-provider API mode.
_ai_client = ai.Client()


def _get_openwebui_client(config: config_lib.Config) -> _OpenAIClient:
    """Return an OpenAI-compatible client pointed at the OpenWebUI instance.

    The base URL is taken from config.openwebui_base_url (which itself
    defaults to the OPENWEBUI_URL env var).
    The API key is read from the OPENWEBUI_API_KEY env var.
    """
    base_url = config.openwebui_base_url.rstrip("/")

    return _OpenAIClient(
        base_url=base_url,
        api_key=os.environ.get("OPENWEBUI_API_KEY", "0"),
    )


def _get_openrouter_client() -> _OpenAIClient:
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    return _OpenAIClient(
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY"),
    )


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """Return multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Sample program skeleton continuations and send them for analysis."""
    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """Continuously sample programs from prompts and send them for analysis."""
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            if DEBUG_PRINTS:
                print(f"\n{'='*80}")
                print(f"Constructing prompt from Island #{prompt.island_id}")
                print(f"{'='*80}")

            reset_time = time.time()
            samples, ok = evaluator.LocalSandbox.safe_run(
                self._llm.draw_samples, prompt.code, self.config)
            if not ok:
                continue
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for idx, sample in enumerate(samples):
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)

                extra_kwargs = {}
                llm_source = getattr(self.config, 'structure_backend', 'local')
                extra_kwargs['llm_source'] = llm_source

                llm_model_name = None
                if llm_source in ('api', 'openwebui'):
                    # Prefer the structure-specific model over the legacy fallback.
                    llm_model_name = getattr(self.config, 'structure_model', None) or getattr(
                        self.config, 'api_model', None
                    )
                else:
                    llm_model_name = getattr(self._llm, 'model_name', None)
                    if llm_model_name is None:
                        llm_model_name = (
                            os.environ.get('LOCAL_LLM_MODEL_PATH')
                            or os.environ.get('LOCAL_LLM_MODEL_NAME')
                        )
                if llm_model_name is not None:
                    extra_kwargs['llm_model_name'] = llm_model_name

                evaluator.LocalSandbox.safe_run(
                    chosen_evaluator.analyse,
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    **extra_kwargs,
                    config=self.config,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """Extract the function body from a model response, preserving indentation."""
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        # API-style backends usually return correctly indented code blocks.
        mode = getattr(config, 'llm_mode', 'local')
        if mode in ('api', 'openwebui'):
            code = ''
            for line in lines[func_body_lineno + 1:]:
                if line and not line[0].isspace() and line.strip():
                    break
                if line.strip().startswith('```'):
                    break
                code += line + '\n'
        else:
            # Local models sometimes omit indentation, so normalize it.
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line and not line[0].isspace() and line.strip():
                    break
                if line.strip().startswith('```'):
                    break
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'

        return code

    return sample


class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        super().__init__(samples_per_prompt)

        self._url = "http://127.0.0.1:5000/completions"

        self._instruction_prompt = (
            "You are a helpful assistant tasked with discovering ordinary differential equations (ODEs) structures for dynamic processes. \
            Complete the 'system' function below, considering the roles of each input, their interactions, and plausible dynamics. \
            Let's think step by step, but provide only (1) a final brief explanation (2-5 sentences) of your reasoning, limitations and unmodeled effects, followed by (2) the completed ODE expressions.\n\n"
        )
        self._batch_inference = batch_inference
        self._trim = trim

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        # Backend for structure inference.
        mode = getattr(config, 'structure_backend', 'local')

        if DEBUG_PRINTS:
            model_name = (
                getattr(config, "structure_model", None)
                or getattr(config, "api_model", None)
                or os.environ.get("LOCAL_LLM_MODEL_NAME")
            )
            print("\n=== [AgentODE] STRUCTURE LLM CONFIG ===")
            print(f"Backend : {mode}")
            print(f"Model   : {model_name}")
            print("=====================================\n")

        if mode == 'api':
            return self._draw_samples_api(prompt, config)
        elif mode == 'openwebui':
            return self._draw_samples_openwebui(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        full_prompt = '\n'.join([self._instruction_prompt, prompt])

        if DEBUG_PRINTS:
            print("=== Prompt being sent to API (aisuite) ===")
            print(full_prompt)
            print("=== End of prompt ===")

        # Prefer structure-specific model if provided; otherwise api_model.
        model_name = getattr(config, "structure_model", None) or getattr(config, "api_model", None)

        provider = str(getattr(config, "api_provider", "openai")).lower()
        api_max_tokens = int(getattr(config, "api_max_tokens", 8192))
        all_samples = []
        for _ in range(self._samples_per_prompt):
            if provider == "openrouter":
                response = _get_openrouter_client().chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=api_max_tokens,
                    extra_headers={
                        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://localhost"),
                        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "AgentODE"),
                    },
                )
            else:
                response = _ai_client.chat.completions.create(
                    model=model_name,  # e.g. "openai:gpt-4o"
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=api_max_tokens,
                )
            content = response.choices[0].message.content

            if DEBUG_PRINTS:
                print("\n----- Raw API Response -----")
                print(content)
                print("----- End Raw API Response -----\n")

            body = _extract_body(content, config) if self._trim else content
            all_samples.append(body)

        return all_samples

    def _draw_samples_openwebui(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        full_prompt = '\n'.join([self._instruction_prompt, prompt])

        if DEBUG_PRINTS:
            print(f"=== Prompt being sent to OpenWebUI ({config.openwebui_base_url}) ===")
            print(full_prompt)
            print("=== End of prompt ===")

        client = _get_openwebui_client(config)
        # Prefer structure-specific model if provided; otherwise api_model.
        model_name = getattr(config, "structure_model", None) or getattr(config, "api_model", None)

        all_samples = []
        api_max_tokens = int(getattr(config, "api_max_tokens", 8192))

        for _ in range(self._samples_per_prompt):
            response = client.chat.completions.create(
                model=model_name,  # e.g. "openai/gpt-5.2-llmlb"
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=api_max_tokens,
            )
            content = response.choices[0].message.content

            if DEBUG_PRINTS:
                print("\n----- Raw OpenWebUI Response -----")
                print(content)
                print("----- End Raw OpenWebUI Response -----\n")

            body = _extract_body(content, config) if self._trim else content
            all_samples.append(body)

        return all_samples

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])

        if DEBUG_PRINTS:
            print("=== Prompt being sent to local LLM ===")
            print(prompt)
            print("=== End of prompt ===")

        while True:
            try:
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        res = self._do_request(prompt)
                        all_samples.append(res)

                if DEBUG_PRINTS:
                    print("\n=== Raw LLM response(s) (pre-trim) ===")
                    for idx, sample in enumerate(all_samples, start=1):
                        print(f"\n----- Raw Sample #{idx} -----")
                        for lineno, line in enumerate(sample.splitlines(), start=1):
                            print(f"{lineno:3}: {line}")
                        print("----- End Raw Sample -----")
                    print("=== End of raw LLM response(s) ===\n")

                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]

                return all_samples

            except Exception as e:
                print(f"[LocalLLM] Error during local sampling: {e}. Retrying in 30 seconds...")
                time.sleep(30)
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'max_new_tokens': 3072,  # local inference engines require this to be set explicitly
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            response = response.json()["content"]
            return response if self._batch_inference else response[0]
