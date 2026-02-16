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

""" Implementation of the LLMODE pipeline. """
from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence

from llmode import code_manipulation
from llmode import config as config_lib
from llmode import evaluator
from llmode import buffer
from llmode import sampler
from llmode import profile
from llmode import param_utils


def _extract_function_names(specification: str) -> Tuple[str, str | None]:
    """Return the name of the function to evolve and of the function to run.

    The specification is the boilerplate code template for a task.
    Historically, the template contained both:
      * `@evaluate.run`  - the evaluation function.
      * `@system.evolve` - the ODE system to be searched.

    For newer problems that use a centralized evaluation (e.g. synthetic
    likelihood), the `@evaluate.run` decorator may be absent. In that case,
    this function returns ``function_to_run = None`` and downstream evaluation
    logic is handled outside the specification.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) == 0:
        function_to_run = None
    elif len(run_functions) == 1:
        function_to_run = run_functions[0]
    else:
        raise ValueError('Expected at most 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'system', 'evolve'))

    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@system.evolve`.')

    return evolve_functions[0], function_to_run



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        problem_name: str,
        **kwargs
):
    """ Launch a LLMODE experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    # Initialise the default system docstring for parameter inference based
    # on the current problem specification, so that downstream prompts
    # reflect the problem-specific Args/Returns block.
    param_utils.set_default_system_docstring_from_program(template, function_to_evolve)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        # Record which LLM/backend and hyperparameters are used for this run
        # in a single metadata file alongside the logs.
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
            function_to_run=function_to_run,
            inputs=inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            problem_name=problem_name,
        ))

    initial = template.get_function(function_to_evolve).body
    # Evaluate the initial specification system once to seed all islands.
    # Treat this as global sample order 0 so subsequent LLM samples start at 1.
    # For centralized evaluation problems, this call may include default
    # parameter priors so that the founder system can be scored.
    founder_param_dists = param_utils.get_default_param_distributions(problem_name)
    evaluators[0].analyse(
        initial,
        island_id=None,
        version_generated=None,
        profiler=profiler,
        problem_name=problem_name,
         config=config,
         global_sample_nums=0,
         sample_time=None,
        param_distributions=founder_param_dists,
        enable_param_optim=False,
    )
    
    # Set global max sample nums.
    samplers = [sampler.Sampler(database, evaluators, 
                                config.samples_per_prompt, 
                                max_sample_nums=max_sample_nums, 
                                llm_class=class_config.llm_class,
                                config = config) 
                                for _ in range(config.num_samplers)]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler, problem_name=problem_name)
