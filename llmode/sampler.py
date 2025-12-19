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

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmode import evaluator
from llmode import buffer
from llmode import config as config_lib
import requests
import json
import http.client
import os



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
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
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            prompt = self._database.get_prompt()
            print(f"\n{'='*80}")
            print(f"Constructing prompt from Island #{prompt.island_id}")
            print(f"{'='*80}")
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for idx, sample in enumerate(samples):
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)

                # Attach parameter distributions and LLM metadata for this sample
                # if available from the LLM / config.
                extra_kwargs = {}
                get_param_distributions = getattr(self._llm, 'get_param_distributions_for_sample', None)
                if callable(get_param_distributions):
                    try:
                        param_distributions = get_param_distributions(idx)
                    except Exception:
                        param_distributions = None
                    if param_distributions is not None:
                        extra_kwargs['param_distributions'] = param_distributions

                # Record which LLM backend/model produced this sample so it can
                # be logged in the profiler JSON.
                llm_source = 'api' if self.config.use_api else 'local'
                extra_kwargs['llm_source'] = llm_source

                llm_model_name = None
                if self.config.use_api:
                    llm_model_name = getattr(self.config, 'api_model', None)
                else:
                    # Prefer a name exposed by the LocalLLM implementation;
                    # fall back to environment variables if unset.
                    llm_model_name = getattr(self._llm, 'model_name', None)
                    if llm_model_name is None:
                        llm_model_name = (
                            os.environ.get('LOCAL_LLM_MODEL_PATH')
                            or os.environ.get('LOCAL_LLM_MODEL_NAME')
                        )
                if llm_model_name is not None:
                    extra_kwargs['llm_model_name'] = llm_model_name

                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    **extra_kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

            # print(f"\n{'='*80}")
            # print(f"After Sample Batch #{self.__class__._global_samples_nums}")
            # print(f"Sampled from Island #{prompt.island_id}, Generated version: v{prompt.version_generated}")
            # print(f"{'='*80}")
            self._database.print_island_info()

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                # Stop at unindented code (like if __name__ or new function definitions)
                if line and not line[0].isspace() and line.strip():
                    break
                # Stop at markdown code blocks
                if line.strip().startswith('```'):
                    break
                code += line + '\n'

        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                # Stop at unindented code or markdown
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
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        # Default engines:
        #   - ODE skeleton generation: port 5000
        #   - Parameter inference:     port 5001 (separate LLM)
        url = "http://127.0.0.1:5000/completions"
        # instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
        #                      Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")

        instruction_prompt = ("You are a biomedical systems modeling expert tasked with discovering ordinary differential equations (ODEs) structures for disease progression. \
                              Complete the incomplete ode system function below, considering the biological roles of each biomarker, their interactions, and plausible pathological dynamics. \
                              Let’s think step by step, but provide only (1) a final brief explanation (2-5 sentences) of your reasoning, followed by (2) the completed ODE expressions.  \n\n")

        self._batch_inference = batch_inference
        self._url = url
        self._param_url = "http://127.0.0.1:5001/completions"
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        self._param_inference_results = None


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])

        print("=== Prompt being sent to LLM ===")
        print(prompt)
        print("=== End of prompt ===")

        # Reset the cached parameter results for this batch.
        self._param_inference_results = None

        while True:
            try:
                all_samples = []
                raw_responses = []
                # response from llm server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        raw_responses.append(res)
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        res = self._do_request(prompt)
                        raw_responses.append(res)
                        all_samples.append(res)

                # Debug: pretty-print raw LLM responses before any trimming
                print("\n=== Raw LLM response(s) (pre-trim) ===")
                # for idx, sample in enumerate(all_samples, start=1):
                #     print(f"\n----- Raw Sample #{idx} -----")
                #     for lineno, line in enumerate(sample.splitlines(), start=1):
                #         print(f"{lineno:3}: {line}")
                #     print("----- End Raw Sample -----")
                # print("=== End of raw LLM response(s) ===\n")

                for idx, sample in enumerate(all_samples, start=1):
                    print(f"\n----- Sample #{idx} (lines={len(sample.splitlines())}) -----")
                    print(sample)
                    print("----- End Sample -----")
                print("=== End of raw LLM response(s) ===\n")

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]

                # print("\n================ TRIMMED CODE SAMPLES ==============")
                # for i, sample in enumerate(all_samples):
                #     print(f"\n--- Sample {i+1} ---\n{sample}")
                # print("====================================================\n")

                # Parameter distribution inference (for local LLM only)
                if config.use_api == False:
                    from llmode import param_utils
                    import json

                    self._param_inference_results = []

                    for i, raw_response in enumerate(raw_responses):
                        try:
                            print(f"\n============ PARAMETER INFERENCE (sample {i + 1}) ==============")
                            # Extract just the function from the raw response
                            function_code = param_utils.extract_function_from_response(raw_response, 'system')
                            # print(f"[Extracted Function]\n{function_code}\n")

                            # Build prompt
                            param_prompt = param_utils.build_param_inference_prompt(
                                system_code=function_code,
                                spec_template_path='specs_parameters/spec_parameters_aki.txt',
                                function_name='system'
                            )
                            # print("[Generated Prompt for Parameter Inference]")
                            # print(param_prompt + "\n")

                            # Send to dedicated parameter-inference LLM
                            # print("[Sending to parameter-inference LLM...]")
                            param_response = self._do_param_request(param_prompt)

                            # print("\n[Raw LLM Response for Parameters]")
                            # print(param_response if isinstance(param_response, str) else param_response[0])
                            # raw_param_text = param_response if isinstance(param_response, str) else param_response[0]

                            # print("\n=== Raw LLM response for parameters ===")
                            # print(f"----- Param Sample #{i + 1} (lines={len(raw_param_text.splitlines())}) -----")
                            # print(raw_param_text)
                            # print("----- End Param Sample -----")
                            # print("=== End of raw parameter response ===\n")


                            # Extract JSON
                            param_json_str = param_response if isinstance(param_response, str) else param_response[0]
                            param_distributions = param_utils.extract_json_from_llm_output(param_json_str)
                            # Enforce that every parameter entry has numeric
                            # mean/sd; if this fails, fall back to defaults.
                            param_distributions = param_utils.validate_param_distributions_format(param_distributions)

                            print("\n[Extracted Parameter Distributions JSON]")
                            print(json.dumps(param_distributions, indent=2))
                            print("====================================================\n")

                            self._param_inference_results.append(param_distributions)

                        except Exception as e:
                            # If anything goes wrong (connection error, bad JSON, etc.),
                            # fall back to problem-level default priors; the evaluator
                            # will fill these in so scoring can proceed.
                            print(f"[Parameter inference failed for sample {i + 1}: {e}]\n")
                            self._param_inference_results.append(None)

                return all_samples
            except Exception as e:
                # If the local LLM server is temporarily unavailable or crashes,
                # wait briefly before retrying to avoid a busy loop.
                print(f"[LocalLLM] Error during ODE/parameter sampling: {e}. Retrying in 5 seconds...")
                time.sleep(30)
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break

                except Exception as e:
                    # Back off on transient API errors instead of tight-looping.
                    print(f"[LocalLLM] Error during API sampling: {e}. Retrying in 5 seconds...")
                    time.sleep(30)
                    continue
        
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'max_new_tokens': 3072,  # Increased from default 512 to allow complete responses
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
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]

    def _do_param_request(self, content: str) -> str:
        """Send a request to the dedicated parameter-inference LLM engine."""
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'max_new_tokens': 3072,
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._param_url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            response = response.json()["content"]
            return response if self._batch_inference else response[0]

    def get_param_distributions_for_sample(self, index: int):
        """Return cached parameter distributions for a given sample index, if available."""
        if self._param_inference_results is None:
            return None
        if 0 <= index < len(self._param_inference_results):
            return self._param_inference_results[index]
        return None
