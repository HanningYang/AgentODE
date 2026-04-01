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

"""A multi-island experience buffer that implements the evolutionary algorithm."""
from __future__ import annotations

from llmode.core import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import os
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

from llmode.core import code_manipulation
from llmode import config as config_lib


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)

        self._last_reset_time: float = time.time()

    def get_best_island_score(
            self,
            island_id: int,
            test_name: str | None = None,
    ) -> float | None:
        """Return the best score for a given island.

        If `test_name` is None, returns the aggregate island score used for
        selection (higher is better). If `test_name` is provided, returns the
        best per-test score for that key (for example, 'euclidean_distance'),
        or None if unavailable.
        """
        if island_id < 0 or island_id >= len(self._islands):
            raise ValueError(f'Invalid island_id: {island_id}')

        if test_name is None:
            best_score = self._best_score_per_island[island_id]
            return None if best_score == -float('inf') else float(best_score)

        scores_per_test = self._best_scores_per_test_per_island[island_id]
        if scores_per_test is None:
            return None
        if test_name not in scores_per_test:
            return None
        return float(scores_per_test[test_name])

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt(island_id=island_id)
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)

        score = round(float(_reduce_score(scores_per_test)), 2)

        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)
            sample_order = getattr(program, 'global_sample_nums', None)
            print(f"[Island {island_id}] Updated best score to {score} (sample {sample_order})")

        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            param_distributions = kwargs.get('param_distributions', None)
            if param_distributions is not None:
                program.param_distributions = param_distributions
            program.score = score
            program.global_sample_nums = kwargs.get('global_sample_nums', None)
            program.sample_time = kwargs.get('sample_time', None)
            program.evaluate_time = kwargs.get('evaluate_time', None)
            llm_source = kwargs.get('llm_source', None)
            llm_model_name = kwargs.get('llm_model_name', None)
            if llm_source is not None:
                program.llm_source = llm_source
            if llm_model_name is not None:
                program.llm_model_name = llm_model_name
            profiler.register_function(program)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)

    def print_island_info(self) -> None:
        """Print detailed information about all islands and their clusters."""
        print("\n" + "="*80)
        print("EXPERIENCE BUFFER - ISLAND & CLUSTER INFORMATION")
        print("="*80)

        for island_id, island in enumerate(self._islands):
            print(f"\n{'─'*80}")
            print(f"ISLAND {island_id}")
            print(f"{'─'*80}")
            print(f"  Best Score: {self._best_score_per_island[island_id]:.6f}")
            print(f"  Total Programs: {island._num_programs}")
            print(f"  Number of Clusters: {len(island._clusters)}")

            if len(island._clusters) > 0:
                print(f"\n  Clusters:")
                for cluster_idx, (signature, cluster) in enumerate(island._clusters.items()):
                    print(f"\n    Cluster {cluster_idx + 1}:")
                    print(f"      Signature: {signature}")
                    print(f"      Score: {cluster.score:.6f}")
                    print(f"      Programs in cluster: {len(cluster._programs)}")
                    print(f"      Program lengths: {cluster._lengths}")
                    sample_nums = [s if s is not None else 0 for s in cluster._sample_nums]
                    print(f"      Sample numbers: {sample_nums}")
            else:
                print(f"  (No clusters yet)")

        print("\n" + "="*80 + "\n")


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self, island_id: int | None = None) -> tuple[str, int]:
        """Constructs a prompt containing system program skeletons from this island."""
        signatures = list(self._clusters.keys())

        if not signatures:
            # No successful programs yet; fall back to the seed program from
            # the template without emitting warnings. In normal runs this
            # path should be unused because the pipeline seeds all islands
            # with a Euclidean-scored founder system.
            seed_func = self._template.get_function(self._function_to_evolve)
            return (str(seed_func), 1)

        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        # Z-score normalize cluster scores before temperature scaling and softmax.
        mean = np.mean(cluster_scores)
        std = np.std(cluster_scores)
        if std > 0:
            normalized_scores = (cluster_scores - mean) / std
        else:
            normalized_scores = np.zeros_like(cluster_scores)
        normalized_scores = np.clip(normalized_scores, -5.0, 5.0)

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        temperature = max(temperature, 0.05)

        probabilities = _softmax(normalized_scores, temperature)
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]

        implementations: list[code_manipulation.Function] = []
        scores: list[float] = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1

        if os.environ.get("LLMODE_DEBUG_PRINTS", "0") == "1":
            sample_orders = [
                getattr(impl, 'global_sample_nums', None)
                for impl in sorted_implementations
            ]
            print(f"  Included sample orders: {sample_orders}")
            print(f"  {'-'*46}")

        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)

        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]
        self._sample_nums: list[int | None] = [
            getattr(implementation, 'global_sample_nums', None)
        ]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))
        self._sample_nums.append(getattr(program, 'global_sample_nums', None))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorter programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
