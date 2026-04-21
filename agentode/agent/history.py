"""Iteration history tracking for the agent workflow.

Maintains per-run iteration history in memory and builds
JSON-serializable summaries of parameter candidates ordered
from worst to best log synthetic likelihood (logSL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math
import json
import os
import time
from pathlib import Path


@dataclass
class ParamCandidate:
    """One parameter set with its associated logSL."""

    log_sl: float
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        stripped_params = {
            k: {"mean": v["mean"], "sd": v["sd"]}
            if isinstance(v, dict) and "mean" in v and "sd" in v
            else v
            for k, v in self.params.items()
        }
        return {
            "log_sl": float(self.log_sl),
            "params": stripped_params,
        }


@dataclass
class IterationEvent:
    """Generic iteration event for future extension."""

    step: int
    kind: str
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "step": self.step,
            "kind": self.kind,
            "timestamp": self.timestamp,
        }
        if self.payload:
            out["payload"] = self.payload
        return out


class IterationHistory:
    """In-memory history for a single optimization run.

    This class is designed to be lightweight and agent-facing:
      - the agent records iteration events as they happen,
      - parameter candidates with valid logSL are tracked,
      - the top-N (configurable) candidates can be retrieved as a
        JSON-ready list ordered from worst to best logSL.
    """

    def __init__(self, max_param_history: int = 3) -> None:
        if max_param_history < 1:
            max_param_history = 1
        self._max_param_history = int(max_param_history)
        self._events: List[IterationEvent] = []
        self._param_candidates: List[ParamCandidate] = []

    @property
    def max_param_history(self) -> int:
        return self._max_param_history

    def record_event(
        self,
        step: int,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a generic iteration event."""
        self._events.append(
            IterationEvent(step=step, kind=kind, payload=payload or {})
        )

    def record_param_candidate(self, log_sl: Optional[float], params: Dict[str, Any]) -> None:
        """Record a parameter candidate if it has a finite logSL.

        The optimizer is expected to treat higher logSL as better.
        Here we simply keep all finite candidates and later sort them.
        """
        if log_sl is None or not math.isfinite(log_sl):
            return
        self._param_candidates.append(ParamCandidate(log_sl=float(log_sl), params=params))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def get_param_history_worst_to_best(self) -> List[Dict[str, Any]]:
        """Return up to `max_param_history` candidates, worst→best by logSL.

        This returns a list of dicts suitable for JSON serialization:
            [
                {"log_sl": -142.3, "params": {...}},
                {"log_sl":  -98.1, "params": {...}},
                {"log_sl":  -87.4, "params": {...}},
            ]
        """
        if not self._param_candidates:
            return []
        # Sort ascending: worst first, best last.
        ordered = sorted(self._param_candidates, key=lambda c: c.log_sl)
        # Keep the top-N best candidates (tail of ascending sort), then
        # present them worst→best so the LLM sees improvement direction.
        best = ordered[-self._max_param_history :]
        return [c.to_dict() for c in best]

    def get_events(self) -> List[Dict[str, Any]]:
        """Return all recorded events as JSON-serializable dicts."""
        return [e.to_dict() for e in self._events]

    def to_json_dict(self) -> Dict[str, Any]:
        """Build a JSON-serializable snapshot of the full history."""
        return {
            "events": self.get_events(),
            "param_history_worst_to_best": self.get_param_history_worst_to_best(),
        }

    def write_json(self, path: str | Path) -> None:
        """Write the iteration history to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)


def build_iteration_index(
    problem_name: str,
    log_path: str,
    sample_number: int,
    summary_max_len: Optional[int] = None,
) -> str:
    """Build a formatted iteration history table for a sample's parameter optimization.

    Reads each ``iter_N`` directory under
    ``workspace/{problem_name}/logs/{basename(log_path)}/sample_{sample_number}/``
    and assembles a table with columns: iter, type, log_sl, summary.

    Args:
        problem_name: Problem name (e.g. ``'aki'``).
        log_path: Log subdirectory name (e.g. ``'aki_0405_harness2'``).
        sample_number: Integer sample index.
        summary_max_len: Maximum characters for the summary column before truncation.

    Returns:
        A formatted multiline string suitable for inserting into
        ``{iteration_index}``.
    """
    sample_dir = os.path.join(
        "workspace", problem_name, "logs", os.path.basename(log_path), f"sample_{sample_number}"
    )

    if not os.path.isdir(sample_dir):
        return f"(no iteration history found at {sample_dir})"

    iter_dirs = sorted(
        (d for d in os.listdir(sample_dir)
         if d.startswith("iter_") and os.path.isdir(os.path.join(sample_dir, d))),
        key=lambda d: int(d.split("_")[1]),
    )

    rows: list[tuple[int, str, str, str]] = []
    for iter_dir in iter_dirs:
        iter_num = int(iter_dir.split("_")[1])
        iter_path = os.path.join(sample_dir, iter_dir)

        params_path = os.path.join(iter_path, "params.json")
        if not os.path.isfile(params_path):
            continue
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        iter_type = params.get("iter_type", "unknown")
        summary = params.get("summary", "")
        if summary_max_len is not None and len(summary) > summary_max_len:
            summary = summary[:summary_max_len].rstrip() + "..."

        log_sl_path = os.path.join(iter_path, "log_sl.json")
        if os.path.isfile(log_sl_path):
            with open(log_sl_path, "r", encoding="utf-8") as f:
                log_sl_data = json.load(f)
            log_sl_val = log_sl_data.get("log_sl")
            log_sl_str = f"{log_sl_val:.1f}" if isinstance(log_sl_val, (int, float)) else "N/A"
        else:
            log_sl_str = "N/A"

        rows.append((iter_num, iter_type, log_sl_str, summary))

    if not rows:
        return "(no iterations recorded)"

    iter_w = max(len("iter"),   *(len(str(r[0])) for r in rows))
    type_w = max(len("type"),   *(len(r[1])      for r in rows))
    sl_w   = max(len("log_sl"), *(len(r[2])      for r in rows))
    sum_w  = max(len("summary"), *(len(r[3])     for r in rows))

    header = (
        f"{'iter':<{iter_w}}   {'type':<{type_w}}   {'log_sl':>{sl_w}}   {'summary':<{sum_w}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for iter_num, iter_type, log_sl_str, summary in rows:
        lines.append(
            f"{iter_num:<{iter_w}}   {iter_type:<{type_w}}   {log_sl_str:>{sl_w}}   {summary:<{sum_w}}"
        )

    return "\n".join(lines)
