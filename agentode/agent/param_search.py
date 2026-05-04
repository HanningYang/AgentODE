"""Parameter-search strategy selection for ODE calibration."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agentode.agent.param_agent import ParameterAgent


class AblationParameterAgent(ParameterAgent):
    """Run repeated initial parameter inference and keep the best logSL."""

    def __init__(
        self,
        config: Any,
        problem_name: str,
        fn_name: str = "system",
        log_dir: Optional[str] = None,
        sample_order: Optional[int] = None,
    ) -> None:
        super().__init__(
            config=config,
            problem_name=problem_name,
            fn_name=fn_name,
            log_dir=log_dir,
            sample_order=sample_order,
        )
        spec_root = getattr(config, "ablation_spec_dir", "specs_params_abla")
        self._spec_path = os.path.join(spec_root, f"spec_params_{problem_name}.txt")
        self._figures_content = ""
        self._figures_bytes_observed = {}

    def run(
        self,
        program_str: str,
        initial_params: Optional[Dict[str, Any]] = None,
        sample_order: Optional[int] = None,
        best_island_score: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        del initial_params
        del best_island_score

        self._setup(program_str)

        try:
            n_trials = max(1, int(getattr(self._config, "ablation_trials", 5)))
        except Exception:
            n_trials = 5

        best_params: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        last_params: Optional[Dict[str, Any]] = None

        for trial_idx in range(n_trials):
            summary, params = self._initial_inference()
            if params is None:
                continue

            last_params = params
            score, _feedback, log_sl_data = self._score(params, sample_order=sample_order)

            trial_prefix = f"[trial {trial_idx + 1}/{n_trials}] "
            if self._logger is not None:
                if score is not None and np.isfinite(score):
                    self._logger.log_initial_inference(
                        trial_prefix + (summary or ""),
                        params,
                        log_sl_data=log_sl_data,
                    )
                else:
                    _text, report_dict = self._implausibility_report(params)
                    self._logger.log_constraint_violation(
                        trial_prefix + (summary or ""),
                        params,
                        report_dict or {},
                    )

            if score is None or not math.isfinite(float(score)):
                print(f"[AblationAgent] sample={sample_order} trial {trial_idx + 1}/{n_trials} logSL=None (implausible)")
                continue

            score_value = float(score)
            if best_score is None or score_value > best_score:
                best_score = score_value
                best_params = params

            print(f"[AblationAgent] sample={sample_order} trial {trial_idx + 1}/{n_trials} logSL={score_value:.3f}  best={best_score:.3f}")

        print(f"[AblationAgent] sample={sample_order} done  best_logSL={best_score:.3f if best_score is not None else 'None'}")

        if best_score is None:
            return last_params, None
        return best_params, best_score


def create_parameter_agent(
    config: Any,
    problem_name: str,
    fn_name: str = "system",
    log_dir: Optional[str] = None,
    sample_order: Optional[int] = None,
) -> ParameterAgent:
    """Instantiate the configured parameter-search strategy."""
    mode = str(getattr(config, "param_search_mode", "default")).lower()
    if mode == "ablation":
        return AblationParameterAgent(
            config=config,
            problem_name=problem_name,
            fn_name=fn_name,
            log_dir=log_dir,
            sample_order=sample_order,
        )
    return ParameterAgent(
        config=config,
        problem_name=problem_name,
        fn_name=fn_name,
        log_dir=log_dir,
        sample_order=sample_order,
    )


__all__ = ["AblationParameterAgent", "create_parameter_agent"]
