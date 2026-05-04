"""Parameter inference agent for ODE system calibration.

Orchestrates the full parameter optimization loop:
  1. Initial LLM inference of parameter distributions.
  2. Constraint-violation repair when trajectories are implausible.
  3. Iterative log-SL update until convergence or patience exhaustion.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from openai import OpenAI as _OpenAIClient

from agentode.agent import prompt_builder
from agentode.agent import violation_check, tool_executor
from agentode.agent.history import IterationHistory, build_iteration_index
from agentode.core import code_manipulation
from agentode.core import param_utils


DEBUG_PRINTS = os.environ.get("AGENTODE_DEBUG_PRINTS", "0") == "1"

_W = 72

def _dbg(label: str, body: str = "") -> None:
    """Print a visually separated debug block."""
    banner = f"{'─' * _W}\n  {label}\n{'─' * _W}"
    print(f"\n{banner}")
    if body:
        print(body)
    print()


class ParameterAgent:
    """Infers and iteratively refines ODE parameter distributions via LLM.

    Usage::

        agent = ParameterAgent(config=config_obj, problem_name="aki")
        best_params, best_score = agent.run(program_str=program)
    """

    def __init__(
        self,
        config: Any,
        problem_name: str,
        fn_name: str = "system",
        log_dir: Optional[str] = None,
        sample_order: Optional[int] = None,
    ) -> None:
        self._config = config
        self._problem_name = problem_name
        self._fn_name = fn_name

        self._log_dir = log_dir
        self._sample_order = sample_order
        self._logger = None
        if log_dir is not None and sample_order is not None:
            from agentode.agent.workspace_logger import WorkspaceLogger
            self._logger = WorkspaceLogger(problem_name, log_dir, sample_order)

        self._spec_path = os.path.join("specs_params", f"spec_params_{problem_name}.txt")
        self._fig_dir = os.path.join("workspace", problem_name, "figures")
        self._figures_content = prompt_builder.build_figures_block_from_dir(self._fig_dir)
        self._figures_bytes_observed: Dict[str, bytes] = {}
        if os.path.isdir(self._fig_dir):
            for fname in os.listdir(self._fig_dir):
                if fname.lower().endswith(".png"):
                    fpath = os.path.join(self._fig_dir, fname)
                    try:
                        with open(fpath, "rb") as f:
                            self._figures_bytes_observed[fname] = f.read()
                    except OSError:
                        continue

        self._system_func: Any = None
        self._system_code_str: str = ""
        self._use_gpu: bool = False
        self._cached_best_params: Optional[Dict[str, Any]] = None
        self._cached_synthetic: Optional[np.ndarray] = None
        self._cached_time_grid: Optional[np.ndarray] = None
        self._cached_ic_config: Optional[Dict[str, Any]] = None
        self._last_invalid_score_details: Optional[Dict[str, Any]] = None

    def run(
        self,
        program_str: str,
        initial_params: Optional[Dict[str, Any]] = None,
        sample_order: Optional[int] = None,
        best_island_score: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Run the full parameter inference loop.

        Args:
            program_str: Full Python program string containing the ODE function.
            initial_params: Previously inferred parameter distributions, if any.
            sample_order: Sample index for logging.
            best_island_score: Best Euclidean score seen on this island, used to
                decide whether to extend the optimization budget.

        Returns:
            (best_params, best_score) where best_score is the log synthetic
            likelihood (higher is better), or None if no valid score was found.
        """
        self._setup(program_str)

        config = self._config
        max_steps, extended_max_steps, patience, rel_thresh = self._parse_optim_config()

        history = IterationHistory()

        params = initial_params
        initial_summary: Optional[str] = None
        if params is None:
            initial_summary, params = self._initial_inference()

        if params is None:
            return None, None

        t0 = time.time()
        score, feedback, init_log_sl_data = self._score(params, sample_order=sample_order)
        _ = time.time() - t0

        best_params = params
        best_score: Optional[float] = None
        best_feedback: Optional[str] = None
        best_implaus: Optional[str] = None
        best_implaus_dict: Optional[Dict] = None
        no_improve_count = 0
        null_score_count = 0
        extended_mode = False
        # If this stays False, every candidate violated constraints.
        ever_valid_logsl = False

        if score is None or not np.isfinite(score):
            best_implaus, best_implaus_dict = self._implausibility_report(params)
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | sample={sample_order} | init logSL=None (implausible)")
        else:
            ever_valid_logsl = True
            best_score = float(score)
            best_feedback = feedback
            history.record_param_candidate(best_score, params)
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | sample={sample_order} | init logSL={best_score:.3f}")

        if self._logger is not None:
            self._logger.log_initial_inference(
                initial_summary,
                params,
                log_sl_data=init_log_sl_data if best_score is not None else None,
                violation_report=best_implaus_dict if best_score is None else None,
            )

        if not extended_mode and best_island_score is not None and extended_max_steps > max_steps:
            extended_mode = self._maybe_extend(
                params, sample_order, best_island_score, max_steps, extended_max_steps
            )
            if extended_mode:
                max_steps = extended_max_steps
                no_improve_count = 0

        steps_done = 1
        while steps_done < max_steps:
            mode = "implausible" if best_score is None else "log_sl"

            if mode == "implausible":
                prompt = self._build_optimization_prompt(
                    mode=mode,
                    current_params=best_params,
                    violation_report=best_implaus,
                    sl_feedback=best_feedback,
                )
                if prompt is None:
                    break
                new_summary, new_params = self._call_llm(prompt)
                new_failure_modes: Optional[Dict] = None
                new_tool_results: Optional[List] = None
            else:
                new_params, new_summary, new_failure_modes, new_tool_results = (
                    self._diagnosis_and_update_step(
                        current_best_params=best_params,
                        history=history,
                    )
                )

            if new_params is None:
                break

            new_score, new_feedback, new_log_sl_data = self._score(new_params, sample_order=sample_order)

            new_implaus: Optional[str] = None
            new_implaus_dict: Optional[Dict] = None
            if new_score is None or not np.isfinite(new_score):
                new_implaus, new_implaus_dict = self._implausibility_report(new_params)

            if self._logger is not None:
                if new_score is not None and np.isfinite(new_score):
                    self._logger.log_normal_iteration(
                        new_summary,
                        new_params,
                        new_log_sl_data or {},
                        new_failure_modes or {"failure_modes": [], "tool_calls": []},
                        new_tool_results or [],
                    )
                else:
                    self._logger.log_constraint_violation(
                        new_summary, new_params, new_implaus_dict or {}
                    )

            if new_score is None or not np.isfinite(new_score):
                null_score_count += 1
                if best_score is None:
                    best_params = new_params
                    best_implaus = new_implaus
                    best_implaus_dict = new_implaus_dict
                    best_feedback = None
                no_improve_count += 1
                if DEBUG_PRINTS:
                    best_str = f"{best_score:.3f}" if best_score is not None else "None"
                    _dbg(
                        f"ParamAgent | sample={sample_order} | step {steps_done+1} | "
                        f"logSL=None  best={best_str}  "
                        f"no_improve={no_improve_count}/{patience}  null_count={null_score_count}/{patience}"
                    )
                if no_improve_count >= patience:
                    break
            else:
                new_score_val = float(new_score)
                ever_valid_logsl = True
                null_score_count = 0
                history.record_param_candidate(new_score_val, new_params)

                if best_score is None:
                    no_improve_count = 0
                else:
                    improvement = new_score_val - best_score
                    if improvement <= 0.0:
                        no_improve_count += 1
                    else:
                        denom = max(abs(best_score) + 1.0, 1.0)
                        rel_improvement = improvement / denom
                        no_improve_count = 0 if rel_improvement >= rel_thresh else no_improve_count + 1

                best_updated = best_score is None or new_score_val > best_score
                if best_updated:
                    best_score = new_score_val
                    best_params = new_params
                    best_feedback = new_feedback
                    best_implaus = None
                    best_implaus_dict = None

                    if not extended_mode and best_island_score is not None and extended_max_steps > max_steps:
                        extended_mode = self._maybe_extend(
                            best_params, sample_order, best_island_score, max_steps, extended_max_steps
                        )
                        if extended_mode:
                            max_steps = extended_max_steps
                            no_improve_count = 0

                if DEBUG_PRINTS:
                    best_str = f"{best_score:.3f}" if best_score is not None else "None"
                    _dbg(
                        f"ParamAgent | sample={sample_order} | step {steps_done+1} | "
                        f"logSL={new_score_val:.3f}  best={best_str}  "
                        f"no_improve={no_improve_count}/{patience}  null_count={null_score_count}/{patience}"
                    )
                if no_improve_count >= patience:
                    break

            steps_done += 1

        # Avoid assigning an MNSD score when every candidate was implausible.
        if not ever_valid_logsl:
            return best_params, None
        return best_params, best_score

    def _setup(self, program_str: str) -> None:
        """Compile program_str → system_func and extract system_code_str."""
        ns: Dict[str, Any] = {}
        exec(program_str, ns)  # noqa: S102
        self._system_func = ns[self._fn_name]

        try:
            prog = code_manipulation.text_to_program(program_str)
            self._system_code_str = str(prog.get_function(self._fn_name))
        except Exception:
            self._system_code_str = ""

        self._use_gpu = self._detect_gpu(self._system_code_str)
        if self._use_gpu:
            try:
                self._system_func._torch_backend = True
            except (AttributeError, TypeError):
                pass

    @staticmethod
    def _detect_gpu(code: str) -> bool:
        lowered = code.lower()
        return "import torch" in lowered or "torch." in lowered

    def _initial_inference(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="INITIAL_INFERENCE",
                placeholders={"ode_system": self._system_code_str},
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to build initial inference prompt: {e}")
            return None, None

        if DEBUG_PRINTS:
            _dbg("PROMPT | INITIAL_INFERENCE", prompt)

        return self._call_llm(prompt, images=self._figures_bytes_observed or None)

    def _build_optimization_prompt(
        self,
        mode: str,
        current_params: Dict[str, Any],
        violation_report: Optional[str],
        sl_feedback: Optional[str],
    ) -> Optional[str]:
        try:
            # Preserve normalized numeric key order from validation.
            current_params_str = json.dumps(current_params or {}, indent=2)
            prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="CONSTRAINT_VIOLATION",
                placeholders={
                    "ode_system": self._system_code_str,
                    "violation_report": violation_report or "",
                    "current_params_json": current_params_str,
                },
            )
            if DEBUG_PRINTS:
                _dbg("PROMPT | CONSTRAINT_VIOLATION", prompt)
            return prompt
        except Exception as e:
            print(f"[ParamAgent] Failed to build optimization prompt (mode={mode}): {e}")
            return None

    def _diagnosis_and_update_step(
        self,
        current_best_params: Dict[str, Any],
        history: IterationHistory,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict], Optional[List]]:
        """Run DIAGNOSIS → tools → UPDATE for a logSL-valid candidate.

        Returns:
            (new_params, summary, failure_modes_data, tool_results)
            where failure_modes_data = {"failure_modes": [...], "tool_calls": [...]}.
        """
        try:
            self._ensure_best_synthetic_cache(current_best_params)
        except Exception as e:
            print(f"[ParamAgent] Failed to generate synthetic data for diagnosis: {e}")
            return None, None, None, None

        try:
            diagnosis_prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="DIAGNOSIS",
                placeholders={
                    "ode_system": self._system_code_str,
                },
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to build DIAGNOSIS prompt: {e}")
            return None, None, None, None

        if DEBUG_PRINTS:
            _dbg("PROMPT | DIAGNOSIS", diagnosis_prompt)

        try:
            diagnosis_figures = self._build_diagnosis_figures_with_timeout()
        except Exception as e:
            print(f"[ParamAgent] Failed to generate diagnosis figures: {e}")
            diagnosis_figures = self._figures_bytes_observed or None

        # Include comparison figures and tool schemas for API/OpenWebUI backends.
        try:
            tool_schemas = prompt_builder.load_tool_schemas(self._problem_name)
        except Exception as e:
            print(f"[ParamAgent] Failed to load tool schemas: {e}")
            tool_schemas = None

        # DIAGNOSIS may span multiple tool-calling turns.
        _MAX_DIAG_TURNS = 5
        current_diag_prompt = diagnosis_prompt
        all_tool_results: List[Dict[str, Any]] = []
        all_failure_modes: List[Dict[str, Any]] = []
        all_tool_calls_logged: List[Dict[str, Any]] = []
        raw_diag: Optional[str] = None

        for _diag_turn in range(_MAX_DIAG_TURNS):
            raw_diag = self._request_llm(
                current_diag_prompt,
                images=diagnosis_figures if _diag_turn == 0 else None,
                tools=tool_schemas or None,
            )
            if not raw_diag:
                return None, None, None, None

            try:
                diag_json = prompt_builder.extract_json_from_llm_output(raw_diag)
            except Exception as e:
                print(f"[ParamAgent] Failed to parse DIAGNOSIS JSON (turn {_diag_turn}): {e}")
                break

            turn_failure_modes = prompt_builder.extract_failure_modes(diag_json)
            all_failure_modes.extend(turn_failure_modes)

            turn_tool_calls = tool_executor.extract_tool_calls(diag_json)
            all_tool_calls_logged.extend(turn_tool_calls)

            if not turn_tool_calls or self._cached_synthetic is None or self._cached_time_grid is None:
                break

            try:
                turn_results = tool_executor.execute_tool_calls(
                    tool_calls=turn_tool_calls,
                    observed=self._cached_synthetic,
                    synthetic=self._cached_synthetic,
                    time_index=self._cached_time_grid,
                    config=self._cached_ic_config or {},
                )
                all_tool_results.extend(turn_results)

                try:
                    turn_table = tool_executor.format_stat_table(
                        problem_name=self._problem_name,
                        tool_results=turn_results,
                    )
                    results_text = f"[Stat comparison (observed vs synthetic) — turn {_diag_turn + 1}]\n{turn_table}"
                except Exception:
                    results_text = json.dumps(turn_results, indent=2, ensure_ascii=False)

                current_diag_prompt = current_diag_prompt + f"\n\n[Tool results]\n{results_text}"

                if DEBUG_PRINTS:
                    _dbg(
                        f"DIAGNOSIS turn={_diag_turn} | tools={[tc.get('tool') for tc in turn_tool_calls]}",
                        results_text,
                    )

            except Exception as e:
                print(f"[ParamAgent] Failed to execute tools (DIAGNOSIS turn {_diag_turn}): {e}")
                break

        failure_modes_data = {"failure_modes": all_failure_modes, "tool_calls": all_tool_calls_logged}
        failure_modes_str = json.dumps(all_failure_modes, indent=2, ensure_ascii=False)

        stat_table_str = ""
        tool_results: List[Dict[str, Any]] = all_tool_results
        if tool_results:
            try:
                stat_table_str = tool_executor.format_stat_table(
                    problem_name=self._problem_name,
                    tool_results=tool_results,
                )
            except Exception as e:
                print(f"[ParamAgent] Failed to format final stat table: {e}")

        iteration_index_str = ""
        if self._log_dir is not None and self._sample_order is not None:
            iteration_index_str = build_iteration_index(
                problem_name=self._problem_name,
                log_path=self._log_dir,
                sample_number=self._sample_order,
            )

        try:
            update_prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="UPDATE",
                placeholders={
                    "ode_system": self._system_code_str,
                    "failure_modes": failure_modes_str,
                    "iteration_index": iteration_index_str,
                },
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to build UPDATE prompt: {e}")
            return None, None, None, None

        if DEBUG_PRINTS:
            _dbg("PROMPT | UPDATE", update_prompt)

        # Send the best iterations separately to avoid bloating the template.
        best_iters = self._read_best_iterations(n=3)
        best_iters_extra: Optional[List[Dict[str, Any]]] = None
        if best_iters:
            best_iters_str = json.dumps(best_iters, indent=2, ensure_ascii=False)
            best_iters_extra = [
                {
                    "role": "user",
                    "content": (
                        "[ITERATION HISTORY — 3 best by logSL, worst→best]\n"
                        + best_iters_str
                    ),
                }
            ]

        raw_update = self._request_llm(update_prompt, extra_messages=best_iters_extra)

        summary, new_params = self._parse_params_response(raw_update)
        return new_params, summary, failure_modes_data, tool_results

    def _score(
        self,
        params: Dict[str, Any],
        sample_order: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[str], Optional[Dict]]:
        """Return (score, feedback, log_sl_data)."""
        from agentode.metrics.synthetic_likelihood import (
            _evaluate_system_logsl_core,
            build_log_sl_json_data,
        )
        log_likelihood, details, s_obs, stat_names, biomarker_names = _evaluate_system_logsl_core(
            system_func=self._system_func,
            problem_name=self._problem_name,
            param_distributions=params,
            verbose=False,
            sample_order=sample_order,
            backend="gpu" if self._use_gpu else "cpu",
        )
        self._last_invalid_score_details = None
        if log_likelihood is None or details is None:
            return None, None, None
        if not np.isfinite(log_likelihood):
            self._last_invalid_score_details = details
        log_sl_data = build_log_sl_json_data(
            log_likelihood, details, s_obs, stat_names, biomarker_names
        )
        return float(log_likelihood), None, log_sl_data

    def _implausibility_report(
        self, params: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Return (text_report, report_dict) for implausible trajectories."""
        try:
            from agentode.ode import initial_condition_utils, ode_simulator

            if self._last_invalid_score_details is not None:
                report_text = self._last_invalid_score_details.get("violation_report_text")
                report_dict = self._last_invalid_score_details.get("violation_report")
                if report_text is not None and report_dict is not None:
                    return str(report_text), report_dict

            ic_config = initial_condition_utils.load_ic_config(self._problem_name)
            n_patients = int(ic_config.get("sample_size", 100))
            param_sets = param_utils.sample_params_from_distributions(
                params,
                n_samples=n_patients,
                random_seed=ic_config.get("random_seed", None),
                distribution="lognormal",
            )
            trajectories, _t, _ic = ode_simulator.simulate_from_config(
                system_func=self._system_func,
                params=param_sets,
                config=ic_config,
                sample_size=n_patients,
                random_seed=ic_config.get("random_seed", None),
            )
            text_report = violation_check.get_unplausible_trajectory_report(
                trajectories=trajectories,
                config=ic_config,
                check_nans=True,
            )
            report_dict = violation_check.compute_violation_report_dict(
                trajectories=trajectories,
                config=ic_config,
                check_nans=True,
            )
            return text_report, report_dict
        except Exception as e:
            print(f"[ParamAgent] Failed to compute implausibility report: {e}")
            return None, None

    def _maybe_extend(
        self,
        params: Dict[str, Any],
        sample_order: Optional[int],
        best_island_score: float,
        current_max: int,
        extended_max: int,
    ) -> bool:
        """Return True if this candidate is competitive enough to extend the budget."""
        try:
            from agentode.metrics import mnsd_score as _mnsd

            dist = _mnsd.evaluate_system_mnsd(
                system_func=self._system_func,
                problem_name=self._problem_name,
                param_distributions=params,
                verbose=False,
                sample_order=sample_order,
                backend="gpu" if self._use_gpu else "cpu",
                standardization=True,
            )
            if dist is not None and np.isfinite(dist):
                score = -float(dist)
                if score >= best_island_score:
                    if DEBUG_PRINTS:
                        _dbg(
                            f"ParamAgent | sample={sample_order} | "
                            f"extending to {extended_max} steps (MNSD={dist:.2f})"
                        )
                    return True
        except Exception as e:
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | L2 extension check failed: {e}")
        return False

    def _request_llm(
        self,
        prompt: str,
        images: Optional[Dict[str, bytes]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Send prompt (and optional images/tools) to the parameter LLM and return raw text.

        Args:
            extra_messages: Optional list of ``{"role": ..., "content": ...}`` dicts
                prepended before the main prompt in the messages array.  For local
                LLM mode (single-string prompt) the content of each message is
                prepended as plain text instead.
        """
        prompt = prompt.strip()
        if not prompt:
            return None

        config = self._config
        mode = getattr(config, "param_backend", None) or getattr(config, "structure_backend", None)
        if mode is None and getattr(config, "use_api", False):
            mode = "api"

        llm_output: Optional[str] = None

        if mode not in ("api", "openwebui"):
            # The local server has no tool API; fold extra messages into the prompt.
            if extra_messages:
                prefix = "\n\n".join(m.get("content", "") for m in extra_messages)
                prompt = prefix + "\n\n" + prompt
            data = {
                "prompt": prompt,
                "repeat_prompt": 1,
                "params": {
                    "max_new_tokens": 3072,
                    "do_sample": True,
                    "temperature": None,
                    "top_k": None,
                    "top_p": None,
                    "add_special_tokens": False,
                    "skip_special_tokens": True,
                },
            }
            try:
                resp = requests.post(
                    "http://127.0.0.1:5001/completions",
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json().get("content")
                llm_output = content if isinstance(content, str) else (content[0] if content else None)
            except Exception as e:
                print(f"[ParamAgent] Local LLM request failed: {e}")
                return None
        else:
            model_name = (
                getattr(config, "param_model", None)
                or getattr(config, "structure_model", None)
                or getattr(config, "api_model", "gpt-3.5-turbo")
            )
            try:
                if mode == "api":
                    llm_output = self._call_api(prompt, model_name, images=images, tools=tools, extra_messages=extra_messages)
                else:
                    llm_output = self._call_openwebui(prompt, model_name, images=images, tools=tools, extra_messages=extra_messages)
            except Exception as e:
                print(f"[ParamAgent] LLM request failed: {e}")
                return None

        if not llm_output:
            return None

        if DEBUG_PRINTS:
            _dbg("LLM RESPONSE", llm_output)

        return llm_output

    def _parse_params_response(
        self,
        llm_output: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse a raw LLM response string into (summary, params).

        Expects ``{"summary": "...", "params": {...}}`` or a bare params dict.
        """
        try:
            raw = prompt_builder.extract_json_from_llm_output(llm_output)
            summary = raw.get("summary", "") if isinstance(raw, dict) else ""
            params_raw = raw.get("params", raw) if isinstance(raw, dict) and "params" in raw else raw
            params = param_utils.validate_param_distributions_format(params_raw)
            return summary, params
        except Exception as e:
            print(f"[ParamAgent] Failed to parse parameter JSON: {e}")
            return None, None

    def _llm_timeout_seconds(self) -> int:
        """Return a bounded timeout for remote LLM requests."""
        try:
            value = int(
                getattr(self._config, "param_llm_timeout_seconds", 0)
                or os.environ.get("AGENTODE_PARAM_LLM_TIMEOUT_SECONDS", "120")
            )
        except Exception:
            value = 300
        return max(1, value)

    def _call_llm(
        self,
        prompt: str,
        images: Optional[Dict[str, bytes]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Send prompt (and optional images/tools) to the parameter LLM.

        Returns:
            (summary, params) where summary is the LLM's free-text reasoning
            and params is the validated parameter distributions dict.
        """
        llm_output = self._request_llm(prompt, images=images, tools=tools)
        if not llm_output:
            return None, None
        return self._parse_params_response(llm_output)

    def _diagnosis_figure_timeout_seconds(self) -> int:
        """Return a bounded timeout for diagnosis figure generation."""
        try:
            value = int(
                getattr(self._config, "param_diag_figure_timeout_seconds", 0)
                or os.environ.get("AGENTODE_PARAM_DIAG_FIGURE_TIMEOUT_SECONDS", "30")
            )
        except Exception:
            value = 30
        return max(1, value)

    def _build_diagnosis_figures_with_timeout(self) -> Optional[Dict[str, bytes]]:
        """Build diagnosis figures, but fall back if plotting takes too long."""
        timeout_s = self._diagnosis_figure_timeout_seconds()

        class _FigureTimeoutError(TimeoutError):
            pass

        def _handle_timeout(_signum, _frame):
            raise _FigureTimeoutError(
                f"diagnosis figure generation exceeded {timeout_s} seconds"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        try:
            return self._build_diagnosis_figures()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    @staticmethod
    def _api_tool_calls_to_json(message_dict: Dict[str, Any]) -> str:
        """Combine text failure_modes and API-level tool_calls into one JSON string.

        When the API returns native tool calls the content field holds the
        failure-modes prose (or JSON), while the actual tool calls live in
        message["tool_calls"].  We merge both into the dict format expected by
        ``extract_failure_modes`` / ``extract_tool_calls``.
        """
        content = message_dict.get("content") or ""
        api_tool_calls = message_dict.get("tool_calls") or []

        failure_modes: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(content) if content.strip().startswith("{") else {}
            failure_modes = parsed.get("failure_modes", [])
        except Exception:
            pass

        tool_calls: List[Dict[str, Any]] = []
        for tc in api_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            tool_calls.append({"tool": name, "arguments": args, "reason": ""})

        return json.dumps({"failure_modes": failure_modes, "tool_calls": tool_calls})

    def _call_api(
        self,
        prompt: str,
        model_name: str,
        images: Optional[Dict[str, bytes]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        provider = getattr(self._config, "api_provider", "openai")
        provider_name = str(provider).lower()
        extra_headers: Dict[str, str] = {}
        if provider_name == "deepseek":
            host, path = "api.deepseek.com", "/v1/chat/completions"
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
        elif provider_name == "openrouter":
            base_host = os.environ.get("OPENROUTER_HOST", "openrouter.ai")
            base_path = os.environ.get("OPENROUTER_PATH", "/api/v1/chat/completions")
            host, path = base_host, base_path
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
            extra_headers = {
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "AgentODE"),
            }
        else:
            host, path = "api.openai.com", "/v1/chat/completions"
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")

        if not api_key:
            raise RuntimeError(f"No API key found for provider '{provider}'.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(extra_headers)

        # DeepSeek rejects image_url blocks, so send plain text there.
        if provider_name == "deepseek":
            content: Any = prompt  # send plain text only
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if images:
                import base64

                for _name, data in images.items():
                    b64 = base64.b64encode(data).decode("ascii")
                    url = f"data:image/png;base64,{b64}"
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )

        if DEBUG_PRINTS:
            num_images = sum(1 for part in content if part.get("type") == "image_url")
            _dbg(f"API REQUEST | provider={provider} | images={num_images}")

        messages: List[Dict[str, Any]] = list(extra_messages or [])
        messages.append({"role": "user", "content": content})
        request_body: Dict[str, Any] = {
            "max_tokens": 8192,
            "model": model_name,
            "messages": messages,
        }
        if tools:
            request_body["tools"] = tools

        payload = json.dumps(request_body)
        conn = http.client.HTTPSConnection(host, timeout=self._llm_timeout_seconds())
        try:
            conn.request("POST", path, payload, headers)
            res = conn.getresponse()
            raw = res.read().decode("utf-8")
        finally:
            conn.close()

        try:
            data = json.loads(raw)
        except Exception:
            raise RuntimeError(f"{provider} API returned non-JSON response: {raw[:200]!r}")

        if "choices" not in data:
            raise RuntimeError(f"{provider} API error response: {data}")

        message = data["choices"][0]["message"]
        if tools and message.get("tool_calls"):
            return self._api_tool_calls_to_json(message)
        return message.get("content")

    def _call_openwebui(
        self,
        prompt: str,
        model_name: str,
        images: Optional[Dict[str, bytes]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        base_url = (
            getattr(self._config, "openwebui_base_url", "").rstrip("/")
            or os.environ.get("OPENWEBUI_URL", "")
        )
        client = _OpenAIClient(
            base_url=base_url,
            api_key=os.environ.get("OPENWEBUI_API_KEY", "0"),
            timeout=self._llm_timeout_seconds(),
        )
        # OpenWebUI accepts the OpenAI chat format.
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            import base64

            for _name, data in images.items():
                b64 = base64.b64encode(data).decode("ascii")
                url = f"data:image/png;base64,{b64}"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

        if DEBUG_PRINTS:
            num_images = sum(1 for part in content if part.get("type") == "image_url")
            _dbg(f"OPENWEBUI REQUEST | url={base_url} | images={num_images}")

        messages: List[Dict[str, Any]] = list(extra_messages or [])
        messages.append({"role": "user", "content": content})
        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "timeout": self._llm_timeout_seconds(),
        }
        if tools:
            create_kwargs["tools"] = tools

        response = client.chat.completions.create(**create_kwargs)
        msg = response.choices[0].message
        if tools and msg.tool_calls:
            raw_tool_calls = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
            return self._api_tool_calls_to_json(
                {"content": msg.content or "", "tool_calls": raw_tool_calls}
            )
        return msg.content

    def _read_best_iterations(self, n: int = 3) -> List[Dict[str, Any]]:
        """Read the top-N iterations by logSL from the sample's log directory.

        Scans all ``iter_N/`` subdirectories, filters to those with a finite
        ``log_sl`` in their ``log_sl.json``, and returns the best N ordered
        worst→best so the LLM can read the improvement direction.

        Each entry contains:
            ``iter``, ``log_sl``, ``log_sl_detail`` (full log_sl.json),
            ``params`` (full params.json).
        """
        if self._log_dir is None or self._sample_order is None:
            return []

        sample_dir = os.path.join(
            "workspace", self._problem_name,
            "logs", os.path.basename(self._log_dir), f"sample_{self._sample_order}",
        )
        if not os.path.isdir(sample_dir):
            return []

        iter_dirs = sorted(
            (d for d in os.listdir(sample_dir)
             if d.startswith("iter_") and os.path.isdir(os.path.join(sample_dir, d))),
            key=lambda d: int(d.split("_")[1]),
        )

        candidates: List[Dict[str, Any]] = []
        for iter_dir in iter_dirs:
            iter_num = int(iter_dir.split("_")[1])
            iter_path = os.path.join(sample_dir, iter_dir)

            log_sl_path = os.path.join(iter_path, "log_sl.json")
            if not os.path.isfile(log_sl_path):
                continue
            try:
                with open(log_sl_path, "r", encoding="utf-8") as f:
                    log_sl_data = json.load(f)
            except Exception:
                continue
            log_sl_val = log_sl_data.get("log_sl")
            if log_sl_val is None or not math.isfinite(float(log_sl_val)):
                continue

            params_path = os.path.join(iter_path, "params.json")
            if not os.path.isfile(params_path):
                continue
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params_data = json.load(f)
            except Exception:
                continue

            candidates.append({
                "iter": iter_num,
                "log_sl": float(log_sl_val),
                "log_sl_detail": log_sl_data,
                "params": params_data,
            })

        best = sorted(candidates, key=lambda c: c["log_sl"])[-n:]
        return best

    def _parse_optim_config(self) -> Tuple[int, int, int, float]:
        """Return (max_steps, extended_max_steps, patience, rel_thresh)."""
        config = self._config
        try:
            base = max(1, int(getattr(config, "param_optim_steps", 1)))
            # In ablation mode, do not silently re-enable optimization.
            if base <= 1:
                extended = base
            else:
                extended = max(base, int(getattr(config, "param_optim_extended_steps", base)))
            patience = max(1, int(getattr(config, "param_optim_patience", 3)))
            rel_thresh = max(0.0, float(getattr(config, "param_optim_rel_improvement", 0.1))) or 0.1
        except Exception:
            base, extended, patience, rel_thresh = 1, 1, 3, 0.1
        return base, extended, patience, rel_thresh

    def _ensure_best_synthetic_cache(self, params: Dict[str, Any]) -> None:
        """Generate and cache valid synthetic trajectories for diagnosis tools."""
        if self._cached_best_params is not None and self._cached_best_params == params:
            return

        from agentode.ode import initial_condition_utils, ode_simulator

        ic_config = initial_condition_utils.load_ic_config(self._problem_name)

        # Match the sampling used by post-hoc evaluation.
        n_patients = 1000
        param_sets = param_utils.sample_params_from_distributions(
            params,
            n_samples=n_patients,
            distribution="lognormal",
        )

        trajectories, time_grid, _ic_dict, _mask, _issues = ode_simulator.simulate_valid_from_config(
            system_func=self._system_func,
            params=param_sets,
            config=ic_config,
            sample_size=param_sets.shape[0],
            random_seed=ic_config.get("random_seed", None),
            check_nans=True,
        )

        self._cached_best_params = params
        self._cached_synthetic = trajectories
        self._cached_time_grid = time_grid
        self._cached_ic_config = ic_config

    def _build_diagnosis_figures(self) -> Optional[Dict[str, bytes]]:
        """Return observed-vs-synthetic comparison figures for diagnosis."""
        if self._cached_synthetic is None or self._cached_time_grid is None or self._cached_ic_config is None:
            return self._figures_bytes_observed or None

        from agentode.agent.figures import generate_observed_vs_synthetic_figures
        from agentode.ode import initial_condition_utils

        observed_names = initial_condition_utils.get_observed_biomarker_order(
            self._cached_ic_config
        )
        all_names = initial_condition_utils.get_biomarker_order(self._cached_ic_config)
        selected_names = [name for name in observed_names if name in all_names]
        observed_indices = [all_names.index(name) for name in selected_names]

        if not observed_indices:
            return self._figures_bytes_observed or None

        trajectories = self._cached_synthetic[..., observed_indices]
        n_patients, n_timepoints, n_vars = trajectories.shape
        if n_patients == 0 or n_timepoints == 0 or n_vars == 0:
            return self._figures_bytes_observed or None

        synth_df = pd.DataFrame(
            {
                "id": np.repeat(np.arange(n_patients), n_timepoints),
                "t": np.tile(self._cached_time_grid.astype(float), n_patients),
            }
        )
        flat = trajectories.reshape(n_patients * n_timepoints, n_vars)
        for idx, name in enumerate(selected_names):
            synth_df[name] = flat[:, idx]

        figures = generate_observed_vs_synthetic_figures(
            problem_name=self._problem_name,
            synthetic_df=synth_df,
            vars=selected_names,
            bin_width=None,  # let IC config supply the default
        )
        return figures or self._figures_bytes_observed or None
