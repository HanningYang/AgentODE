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

""" Implementation of the AgentODE pipeline. """
from __future__ import annotations

from typing import Any, Tuple, Sequence
import copy
import time

import numpy as np

from agentode.core import code_manipulation
from agentode import config as config_lib
from agentode.core import evaluator
from agentode.core import buffer
from agentode.core import sampler
from agentode.core import profile
from agentode.core import param_utils
from agentode.core import checkpoint as checkpoint_lib
from agentode.metrics import mntd_score as _mntd_score


def _extract_function_name(specification: str) -> str:
    """Return the name of the function decorated with `@system.evolve`."""
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, 'system', 'evolve')
    )

    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@system.evolve`.')

    return evolve_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        problem_name: str,
        **kwargs
):
    """ Launch a AgentODE experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve = _extract_function_name(specification)
    template = code_manipulation.text_to_program(specification)

    param_utils.set_default_system_docstring_from_program(template, function_to_evolve)

    log_dir = kwargs.get('log_dir', None)

    # --- Checkpoint: try to resume from a previous run in log_dir ---
    resumed_database, resumed_sample_num = (None, None)
    if log_dir is not None:
        resumed_database, resumed_sample_num = checkpoint_lib.load(log_dir)

    if resumed_database is not None:
        database = resumed_database
        # Restore the template reference so new Islands use the current spec.
        database._template = template
        sampler.Sampler._global_samples_nums = resumed_sample_num
        already_seeded = True
    else:
        database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)
        already_seeded = False

    # Wrap register_program to save a checkpoint after every registration.
    if log_dir is not None:
        _orig_register = database.register_program

        def _register_with_checkpoint(program, island_id, scores_per_test, **kw):
            _orig_register(program, island_id, scores_per_test, **kw)
            checkpoint_lib.save(
                log_dir, database,
                sampler.Sampler._global_samples_nums,
            )

        database.register_program = _register_with_checkpoint

    if log_dir is None:
        profiler = None
    else:
        llm_metadata = None
        get_run_metadata = getattr(class_config.llm_class, 'get_run_metadata', None)
        if callable(get_run_metadata):
            try:
                llm_metadata = get_run_metadata(config)
            except Exception:
                llm_metadata = None

        profiler = profile.Profiler(log_dir, llm_metadata=llm_metadata)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database=database,
            template=template,
            function_to_evolve=function_to_evolve,
            inputs=inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            problem_name=problem_name,
            log_dir=log_dir,
        ))
    # Seed all islands with the Euclidean-scored template (sample 0).
    # Skip when resuming — islands already contain programs from the checkpoint.
    if already_seeded:
        print('[Pipeline] Resumed from checkpoint — skipping island seeding.')
    else:
        try:
            founder_func = copy.deepcopy(template.get_function(function_to_evolve))

            system_code_str = str(founder_func)
            lowered = system_code_str.lower()
            ns: dict[str, Any] = {"np": np}
            if "torch" in lowered:
                import torch as _torch  # type: ignore[import-not-found]
                ns["torch"] = _torch
            exec(system_code_str, ns)  # noqa: S102
            system_func = ns[function_to_evolve]

            use_gpu_backend = ("import torch" in lowered) or ("torch." in lowered)

            founder_param_dists = param_utils.get_default_param_distributions(problem_name)

            t0 = time.time()
            score_value = None
            try:
                from agentode.agent.param_agent import ParameterAgent
                agent = ParameterAgent(
                    config=config,
                    problem_name=problem_name,
                    log_dir=log_dir,
                    sample_order=0,
                )
                final_params, _ = agent.run(
                    program_str=str(template),
                    initial_params=founder_param_dists,
                    sample_order=0,
                    best_island_score=None,
                )
                if final_params is not None:
                    founder_param_dists = final_params
                    dist = _mntd_score.evaluate_system_mntd(
                        system_func=system_func,
                        problem_name=problem_name,
                        param_distributions=final_params,
                        verbose=False,
                        sample_order=0,
                        backend="gpu" if use_gpu_backend else "cpu",
                        standardization=True,
                    )
                    if dist is not None and np.isfinite(dist):
                        score_value = -round(float(dist), 2)
            except Exception as e:
                print(f"[Pipeline] ParameterAgent failed for template, falling back to default params: {e}")
                if founder_param_dists is not None:
                    dist = _mntd_score.evaluate_system_mntd(
                        system_func=system_func,
                        problem_name=problem_name,
                        param_distributions=founder_param_dists,
                        verbose=False,
                        sample_order=0,
                        backend="gpu" if use_gpu_backend else "cpu",
                        standardization=True,
                    )
                    if dist is not None and np.isfinite(dist):
                        score_value = -round(float(dist), 2)
            eval_time = time.time() - t0

            if score_value is None:
                score_value = 0.0

            founder_func.global_sample_nums = 0
            founder_func.sample_time = None
            founder_func.evaluate_time = eval_time
            founder_func.param_distributions = founder_param_dists
            founder_func.score = score_value

            if score_value is not None:
                scores_per_test = {"mntd_score": score_value}
                database.register_program(
                    founder_func,
                    island_id=None,  # seed *all* islands with the template
                    scores_per_test=scores_per_test,
                    profiler=profiler,
                    param_distributions=founder_param_dists,
                    global_sample_nums=0,
                    sample_time=None,
                    evaluate_time=eval_time,
                    llm_source="template",
                    llm_model_name=None,
                )
            elif profiler:
                profiler.register_function(founder_func)
        except Exception as e:
            print(f"[Pipeline] Failed to seed islands with template: {e}")

    samplers = [sampler.Sampler(database, evaluators,
                                config.samples_per_prompt,
                                max_sample_nums=max_sample_nums,
                                llm_class=class_config.llm_class,
                                config=config)
                for _ in range(config.num_samplers)]

    for s in samplers:
        s.sample(profiler=profiler, problem_name=problem_name)
