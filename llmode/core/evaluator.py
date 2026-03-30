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

from __future__ import annotations

from abc import abstractmethod, ABC
import os
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
from llmode.core import profile
import multiprocessing

import numpy as np

from llmode.core import code_manipulation
from llmode.core import buffer
from llmode.core import evaluator_accelerate


DEBUG_PRINTS = os.environ.get("LLMODE_DEBUG_PRINTS", "0") == "1"

class _FunctionLineVisitor(ast.NodeVisitor):

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        assert self._function_end_line is not None
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    # Indentation of generated_code is REQUIRED.
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:

    def _strip_leading_docstring(body: str, keep_docstring: bool) -> str:
        if not keep_docstring or not body:
            return body

        lines = body.splitlines()
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            return body

        stripped = lines[i].lstrip()
        if stripped.startswith('"""'):
            quote = '"""'
        elif stripped.startswith("'''"):
            quote = "'''"
        else:
            return body

        if stripped.count(quote) >= 2:
            i += 1
        else:
            i += 1
            while i < len(lines) and quote not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            return ''
        return '\n'.join(lines[i:]) + '\n\n'

    body = _trim_function_body(generated_code)

    try:
        template_func = template.get_function(function_to_evolve)
        has_template_doc = bool((template_func.docstring or '').strip())
    except ValueError:
        has_template_doc = False

    if has_template_doc:
        body = _strip_leading_docstring(body, keep_docstring=True)

    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body

    return evolved_function, str(program)


class Sandbox(ABC):

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):

    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str,
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:

        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            results = None, False
        else:
            results = self._get_results(result_queue)

        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)

        return results

        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'system'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')


    def _compile_and_run_function(self, program, function_to_run, function_to_evolve,
                                  dataset, numba_accelerate, result_queue):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )

            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)

            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))

        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, False))


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str,
            inputs: Sequence[Any],
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox,
            problem_name: str | None = None,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()  # retained for compatibility, unused
        self._problem_name = problem_name

    @staticmethod
    def _is_torch_system(system_code: str) -> bool:
        lowered = system_code.lower()
        return ("import torch" in lowered) or ("torch." in lowered)

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs
    ) -> None:
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)

        global_sample_nums = kwargs.get('global_sample_nums', None)
        sample_time = kwargs.get('sample_time', None)
        if global_sample_nums is not None:
            new_function.global_sample_nums = global_sample_nums
        if sample_time is not None:
            new_function.sample_time = sample_time

        param_distributions = kwargs.pop('param_distributions', None)
        if param_distributions is not None:
            kwargs['param_distributions'] = param_distributions

        enable_param_optim = kwargs.pop('enable_param_optim', True)

        scores_per_test = {}
        new_function.param_distributions = param_distributions

        time_reset = time.time()
        sample_order = getattr(new_function, 'global_sample_nums', None)

        # Centralized evaluation: parameter optimization via ParameterAgent,
        # followed by Euclidean-distance scoring in summary-stat space.
        try:
            from llmode.agent.param_agent import ParameterAgent
            from llmode.metrics import euclidean_score as _euclidean_score

            problem_name = kwargs.get('problem_name') or self._problem_name
            if problem_name is None:
                raise ValueError(
                    'Centralized evaluation requires `problem_name` to be provided.'
                )

            config_obj = kwargs.get('config', None)

            best_island_euclid_score: float | None = None
            if (
                enable_param_optim
                and isinstance(self._database, buffer.ExperienceBuffer)
                and island_id is not None
            ):
                try:
                    best_island_euclid_score = self._database.get_best_island_score(
                        island_id,
                        test_name='euclidean_distance',
                    )
                except Exception:
                    best_island_euclid_score = None

            agent = ParameterAgent(config=config_obj, problem_name=problem_name)
            final_params, final_score = agent.run(
                program_str=program,
                initial_params=param_distributions if enable_param_optim else None,
                sample_order=sample_order,
                best_island_score=best_island_euclid_score if enable_param_optim else None,
            )

            param_distributions = final_params
            kwargs['param_distributions'] = final_params
            new_function.param_distributions = final_params

            if final_params is not None:
                try:
                    # Re-compile to get system_func for Euclidean scoring.
                    _ns: dict[str, Any] = {}
                    exec(program, _ns)
                    system_func = _ns[self._function_to_evolve]
                    use_gpu_backend = self._is_torch_system(
                        str(code_manipulation.text_to_program(program)
                            .get_function(self._function_to_evolve))
                    )
                    dist = _euclidean_score.evaluate_system_euclidean_distance(
                        system_func=system_func,
                        problem_name=problem_name,
                        param_distributions=final_params,
                        verbose=False,
                        sample_order=sample_order,
                        backend="gpu" if use_gpu_backend else "cpu",
                        standardization=True,
                    )
                    if dist is not None and np.isfinite(dist):
                        scores_per_test['euclidean_distance'] = -round(float(dist), 2)
                except Exception as e:
                    print(f"[Centralized evaluation] Failed to compute Euclidean distance: {e}")
        except Exception as e:
            prefix = f"Sample {sample_order}: " if sample_order is not None else ""
            print(f"{prefix}[Centralized evaluation] Failed with error: {e}")

        evaluate_time = time.time() - time_reset

        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time,
            )

        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                llm_source = kwargs.get('llm_source', None)
                llm_model_name = kwargs.get('llm_model_name', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                if llm_source is not None:
                    new_function.llm_source = llm_source
                if llm_model_name is not None:
                    new_function.llm_model_name = llm_model_name
                profiler.register_function(new_function)
