"""Parameter inference agent for ODE system calibration.

Orchestrates the full parameter optimization loop:
  1. Initial LLM inference of parameter distributions.
  2. Constraint-violation repair when trajectories are implausible.
  3. Iterative log-SL update until convergence or patience exhaustion.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from openai import OpenAI as _OpenAIClient

from llmode.agent import prompt_builder
from llmode.agent import violation_check, tool_executor
from llmode.agent import figures as _figures
from llmode.agent.history import IterationHistory
from llmode.core import code_manipulation
from llmode.core import param_utils


DEBUG_PRINTS = os.environ.get("LLMODE_DEBUG_PRINTS", "0") == "1"

_W = 72  # banner width

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
    ) -> None:
        self._config = config
        self._problem_name = problem_name
        self._fn_name = fn_name

        self._spec_path = os.path.join("specs_params", f"spec_params_{problem_name}.txt")
        self._fig_dir = os.path.join("workspace", problem_name, "figures")
        self._figures_content = prompt_builder.build_figures_block_from_dir(self._fig_dir)
        # Load observed figures as PNG bytes
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

        # Set per-run in _setup().
        self._system_func: Any = None
        self._system_code_str: str = ""
        self._use_gpu: bool = False
        # Cached synthetic data and figures for the current best params.
        self._cached_best_params: Optional[Dict[str, Any]] = None
        self._cached_synthetic: Optional[np.ndarray] = None
        self._cached_time_grid: Optional[np.ndarray] = None
        self._cached_ic_config: Optional[Dict[str, Any]] = None
        self._cached_figures_bytes: Dict[str, bytes] = {}
        self._cached_figures_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # ---- Step 1: initial parameter inference --------------------------------
        params = initial_params
        if params is None:
            params = self._initial_inference()

        if params is None:
            # No valid params at all — nothing to optimize.
            return None, None

        # ---- Step 2: evaluate initial params ------------------------------------
        t0 = time.time()
        score, feedback = self._score(params, sample_order=sample_order)
        _ = time.time() - t0

        best_params = params
        best_score: Optional[float] = None
        best_feedback: Optional[str] = None
        best_implaus: Optional[str] = None
        no_improve_count = 0
        null_score_count = 0
        extended_mode = False

        if score is None or not np.isfinite(score):
            best_implaus = self._implausibility_report(params)
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | sample={sample_order} | init logSL=None (implausible)")
        else:
            best_score = float(score)
            best_feedback = feedback
            history.record_param_candidate(best_score, params)
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | sample={sample_order} | init logSL={best_score:.3f}")

        # Check early extension based on initial L2 score.
        if not extended_mode and best_island_score is not None and extended_max_steps > max_steps:
            extended_mode = self._maybe_extend(
                params, sample_order, best_island_score, max_steps, extended_max_steps
            )
            if extended_mode:
                max_steps = extended_max_steps

        # ---- Step 3: optimization loop ------------------------------------------
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
                new_params = self._call_llm(prompt)
            else:
                new_params = self._diagnosis_and_update_step(
                    current_best_params=best_params,
                    history=history,
                )

            if new_params is None:
                break

            new_score, new_feedback = self._score(new_params, sample_order=sample_order)

            if new_score is None or not np.isfinite(new_score):
                new_implaus = self._implausibility_report(new_params)
                null_score_count += 1
                if best_score is None:
                    best_params = new_params
                    best_implaus = new_implaus
                    best_feedback = None
                    # First None gets a grace period; from the second onwards
                    # it counts as no improvement.
                    if null_score_count > 1:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                if DEBUG_PRINTS:
                    best_str = f"{best_score:.3f}" if best_score is not None else "None"
                    _dbg(
                        f"ParamAgent | sample={sample_order} | step {steps_done+1} | "
                        f"logSL=None  best={best_str}  "
                        f"no_improve={no_improve_count}/{patience}  null_count={null_score_count}/{patience}"
                    )
                if null_score_count >= patience:
                    break
            else:
                new_score_val = float(new_score)
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

                    if not extended_mode and best_island_score is not None and extended_max_steps > max_steps:
                        extended_mode = self._maybe_extend(
                            best_params, sample_order, best_island_score, max_steps, extended_max_steps
                        )
                        if extended_mode:
                            max_steps = extended_max_steps

                if DEBUG_PRINTS:
                    best_str = f"{best_score:.3f}" if best_score is not None else "None"
                    _dbg(
                        f"ParamAgent | sample={sample_order} | step {steps_done+1} | "
                        f"logSL={new_score_val:.3f}  best={best_str}  "
                        f"no_improve={no_improve_count}/{patience}  null_count={null_score_count}/{patience}"
                    )
                if best_score is not None and no_improve_count >= patience:
                    break

            steps_done += 1

        return best_params, best_score

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Initial inference
    # ------------------------------------------------------------------

    def _initial_inference(self) -> Optional[Dict[str, Any]]:
        try:
            prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="INITIAL_INFERENCE",
                placeholders={"ode_system": self._system_code_str},
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to build initial inference prompt: {e}")
            return None

        if DEBUG_PRINTS:
            _dbg("PROMPT | INITIAL_INFERENCE", prompt)

        # For API/OpenWebUI backends, pass observed figures as images.
        return self._call_llm(prompt, images=self._figures_bytes_observed or None)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_optimization_prompt(
        self,
        mode: str,
        current_params: Dict[str, Any],
        violation_report: Optional[str],
        sl_feedback: Optional[str],
    ) -> Optional[str]:
        try:
            current_params_str = json.dumps(current_params or {}, indent=2, sort_keys=True)
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

    # ------------------------------------------------------------------
    # Diagnosis + update for logSL-valid candidates
    # ------------------------------------------------------------------

    def _diagnosis_and_update_step(
        self,
        current_best_params: Dict[str, Any],
        history: IterationHistory,
    ) -> Optional[Dict[str, Any]]:
        """Run DIAGNOSIS → tools → UPDATE for a logSL-valid candidate."""
        # Ensure synthetic data and comparison figures are cached for current best params.
        try:
            self._ensure_best_synthetic_cache(current_best_params)
        except Exception as e:
            print(f"[ParamAgent] Failed to generate synthetic data for diagnosis: {e}")
            return None

        iteration_history = history.get_param_history_worst_to_best()
        iteration_history_str = json.dumps(iteration_history, indent=2, ensure_ascii=False)

        # 1) Build DIAGNOSIS prompt.
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
            return None

        if DEBUG_PRINTS:
            _dbg("PROMPT | DIAGNOSIS", diagnosis_prompt)

        # For DIAGNOSIS, include comparison figures (observed vs synthetic)
        # when using API/OpenWebUI so the LLM can inspect actual plots.
        # Also pass the tool schemas so the model uses the correct tool names.
        try:
            tool_schemas = prompt_builder.load_tool_schemas(self._problem_name)
        except Exception as e:
            print(f"[ParamAgent] Failed to load tool schemas: {e}")
            tool_schemas = None

        raw_diag = self._request_llm(
            diagnosis_prompt,
            images=self._cached_figures_bytes or None,
            tools=tool_schemas or None,
        )
        if not raw_diag:
            return None

        # if DEBUG_PRINTS:
            # print("[ParamAgent] Raw DIAGNOSIS LLM response:\n", raw_diag)

        try:
            diag_json = prompt_builder.extract_json_from_llm_output(raw_diag)
        except Exception as e:
            print(f"[ParamAgent] Failed to parse DIAGNOSIS JSON: {e}")
            return None

        failure_modes = prompt_builder.extract_failure_modes(diag_json)
        tool_calls = tool_executor.extract_tool_calls(diag_json)

        failure_modes_str = json.dumps(failure_modes, indent=2, ensure_ascii=False)

        # 2) Execute requested tools and build stat table.
        stat_table_str = ""
        if tool_calls and self._cached_synthetic is not None and self._cached_time_grid is not None:
            try:
                # Use synthetic as both observed and synthetic here; observed values
                # in the comparison table are taken from ts_stats.json.
                tool_results = tool_executor.execute_tool_calls(
                    tool_calls=tool_calls,
                    observed=self._cached_synthetic,
                    synthetic=self._cached_synthetic,
                    time_index=self._cached_time_grid,
                    config=self._cached_ic_config or {},
                )
                stat_table_str = tool_executor.format_stat_table(
                    problem_name=self._problem_name,
                    tool_results=tool_results,
                )
            except Exception as e:
                print(f"[ParamAgent] Failed to execute tools or format stat table: {e}")

        # 3) Build UPDATE prompt and request new parameter distributions.
        try:
            update_prompt = prompt_builder.build_param_prompt(
                spec_path=self._spec_path,
                prompt_type="UPDATE",
                placeholders={
                    "ode_system": self._system_code_str,
                    "failure_modes": failure_modes_str,
                    "stat_table": stat_table_str,
                    "iteration_history": iteration_history_str,
                },
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to build UPDATE prompt: {e}")
            return None

        if DEBUG_PRINTS:
            _dbg("PROMPT | UPDATE", update_prompt)

        return self._call_llm(update_prompt)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        params: Dict[str, Any],
        sample_order: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[str]]:
        from llmode.metrics import synthetic_likelihood
        score, feedback = synthetic_likelihood.evaluate_system_logsl_with_feedback(
            system_func=self._system_func,
            problem_name=self._problem_name,
            param_distributions=params,
            verbose=False,
            sample_order=sample_order,
            backend="gpu" if self._use_gpu else "cpu",
        )
        return score, feedback

    def _implausibility_report(self, params: Dict[str, Any]) -> Optional[str]:
        try:
            from llmode.ode import initial_condition_utils, ode_simulator

            means, _ = param_utils.param_distributions_to_arrays(params)
            ic_config = initial_condition_utils.load_ic_config(self._problem_name)
            trajectories, _t, _ic = ode_simulator.simulate_from_config(
                system_func=self._system_func,
                params=means,
                config=ic_config,
                sample_size=None,
                random_seed=ic_config.get("random_seed", None),
            )
            return violation_check.get_unplausible_trajectory_report(
                trajectories=trajectories,
                config=ic_config,
                check_nans=True,
            )
        except Exception as e:
            print(f"[ParamAgent] Failed to compute implausibility report: {e}")
            return None

    # ------------------------------------------------------------------
    # Extension logic
    # ------------------------------------------------------------------

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
            from llmode.metrics import euclidean_score as _euclid

            dist = _euclid.evaluate_system_euclidean_distance(
                system_func=self._system_func,
                problem_name=self._problem_name,
                param_distributions=params,
                verbose=False,
                sample_order=sample_order,
                backend="gpu" if self._use_gpu else "cpu",
                standardization=True,
            )
            if dist is not None and np.isfinite(dist):
                l2 = -float(dist)
                if l2 >= best_island_score:
                    if DEBUG_PRINTS:
                        _dbg(f"ParamAgent | sample={sample_order} | extending to {extended_max} steps (L2={dist:.2f})")
                    return True
        except Exception as e:
            if DEBUG_PRINTS:
                _dbg(f"ParamAgent | L2 extension check failed: {e}")
        return False

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    def _request_llm(
        self,
        prompt: str,
        images: Optional[Dict[str, bytes]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Send prompt (and optional images/tools) to the parameter LLM and return raw text."""
        prompt = prompt.strip()
        if not prompt:
            return None

        config = self._config
        mode = getattr(config, "param_backend", None) or getattr(config, "structure_backend", None)
        if mode is None and getattr(config, "use_api", False):
            mode = "api"

        llm_output: Optional[str] = None

        if mode not in ("api", "openwebui"):
            # Local parameter LLM server on port 5001. Tools are not supported
            # by the local server; ignore the tools argument.
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
                    llm_output = self._call_api(prompt, model_name, images=images, tools=tools)
                else:
                    llm_output = self._call_openwebui(prompt, model_name, images=images, tools=tools)
            except Exception as e:
                print(f"[ParamAgent] LLM request failed: {e}")
                return None

        if not llm_output:
            return None

        if DEBUG_PRINTS:
            _dbg("LLM RESPONSE", llm_output)

        return llm_output

    def _call_llm(
        self,
        prompt: str,
        images: Optional[Dict[str, bytes]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send prompt (and optional images) to the parameter LLM and return parsed distributions."""
        llm_output = self._request_llm(prompt, images=images)
        if not llm_output:
            return None

        try:
            raw = prompt_builder.extract_json_from_llm_output(llm_output)
            return param_utils.validate_param_distributions_format(raw)
        except Exception as e:
            print(f"[ParamAgent] Failed to parse parameter JSON: {e}")
            return None

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

        # Try to extract failure_modes from text content.
        failure_modes: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(content) if content.strip().startswith("{") else {}
            failure_modes = parsed.get("failure_modes", [])
        except Exception:
            pass

        # Convert API-format tool calls to our internal dict format.
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
    ) -> Optional[str]:
        provider = getattr(self._config, "api_provider", "openai")
        if str(provider).lower() == "deepseek":
            host, path = "api.deepseek.com", "/chat/completions"
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
        else:
            host, path = "api.openai.com", "/v1/chat/completions"
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")

        if not api_key:
            raise RuntimeError(f"No API key found for provider '{provider}'.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build multimodal content if images are provided; otherwise plain text.
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

        request_body: Dict[str, Any] = {
            "max_tokens": 8192,
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
        }
        if tools:
            request_body["tools"] = tools

        payload = json.dumps(request_body)
        conn = http.client.HTTPSConnection(host)
        conn.request("POST", path, payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
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
    ) -> Optional[str]:
        base_url = (
            getattr(self._config, "openwebui_base_url", "").rstrip("/")
            or os.environ.get("OPENWEBUI_URL", "https://openwebui.uni-freiburg.de/api")
        )
        client = _OpenAIClient(
            base_url=base_url,
            api_key=os.environ.get("OPENWEBUI_API_KEY", "0"),
        )
        # OpenWebUI speaks the OpenAI chat API; send multimodal content if
        # images are available, otherwise plain text.
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

        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
        }
        if tools:
            create_kwargs["tools"] = tools

        response = client.chat.completions.create(**create_kwargs)
        msg = response.choices[0].message
        if tools and msg.tool_calls:
            # Convert SDK tool call objects to plain dicts for _api_tool_calls_to_json.
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

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _parse_optim_config(self) -> Tuple[int, int, int, float]:
        """Return (max_steps, extended_max_steps, patience, rel_thresh)."""
        config = self._config
        try:
            base = max(1, int(getattr(config, "param_optim_steps", 1)))
            extended = max(base, int(getattr(config, "param_optim_extended_steps", base)))
            patience = max(1, int(getattr(config, "param_optim_patience", 3)))
            rel_thresh = max(0.0, float(getattr(config, "param_optim_rel_improvement", 0.1))) or 0.1
        except Exception:
            base, extended, patience, rel_thresh = 1, 1, 3, 0.1
        return base, extended, patience, rel_thresh

    # ------------------------------------------------------------------
    # Synthetic cache helpers
    # ------------------------------------------------------------------

    def _ensure_best_synthetic_cache(self, params: Dict[str, Any]) -> None:
        """Generate and cache valid synthetic trajectories and comparison figures."""
        if self._cached_best_params is not None and self._cached_best_params == params:
            return

        from llmode.ode import initial_condition_utils, ode_simulator
        import pandas as pd

        ic_config = initial_condition_utils.load_ic_config(self._problem_name)

        # Sample per-patient parameters from the current distributions,
        # mirroring the behaviour used in post-hoc evaluation utilities.
        n_patients = 1000
        param_sets = param_utils.sample_params_from_distributions(
            params,
            n_samples=n_patients,
            distribution="lognormal",
        )

        # Generate trajectories and drop invalid ones.
        trajectories, time_grid, _ic_dict, _mask, _issues = ode_simulator.simulate_valid_from_config(
            system_func=self._system_func,
            params=param_sets,
            config=ic_config,
            sample_size=param_sets.shape[0],
            random_seed=ic_config.get("random_seed", None),
            check_nans=True,
        )

        # Build long-format synthetic DataFrame with columns id, t, biomarkers.
        biomarker_names: List[str] = initial_condition_utils.get_biomarker_order(ic_config)
        records: List[Dict[str, Any]] = []
        n_valid, n_time, _ = trajectories.shape
        for pid in range(n_valid):
            for t_idx in range(n_time):
                row: Dict[str, Any] = {
                    "id": int(pid),
                    "t": float(time_grid[t_idx]),
                }
                for j, name in enumerate(biomarker_names):
                    row[name] = float(trajectories[pid, t_idx, j])
                records.append(row)

        synth_df = pd.DataFrame(records)

        # Generate comparison figures (PNG bytes) from observed vs synthetic.
        figures_bytes = _figures.generate_observed_vs_synthetic_figures(
            problem_name=self._problem_name,
            synthetic_df=synth_df,
        )

        # Simple textual description for the prompt placeholder.
        fig_keys = sorted(figures_bytes.keys())
        figures_text = "\n".join(fig_keys)

        self._cached_best_params = params
        self._cached_synthetic = trajectories
        self._cached_time_grid = time_grid
        self._cached_ic_config = ic_config
        self._cached_figures_bytes = figures_bytes
        self._cached_figures_text = figures_text
