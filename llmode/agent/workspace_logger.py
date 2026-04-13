"""Per-sample workspace logger for parameter optimization iterations.

Directory layout::

    workspace/<problem_name>/<log_dir>/sample_<N>/iter_<N>/<files>

Each iteration produces a ``params.json`` plus type-specific files:

* ``initial_inference`` → ``params.json`` only
* ``constraint_violation`` → ``params.json`` + ``violation_report.json``
* ``normal`` → ``params.json`` + ``log_sl.json`` + ``failure_modes.json``
  + ``tool_results.json``
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


_LOG_SL_COMPONENT_DESCRIPTIONS: Dict[str, Any] = {
    "per_biomarker": [
        {
            "name": "quantile_poly_deg0",
            "description": "intercept of linear fit to empirical quantile function of z-scored values, where z-scoring uses the observed dataset mean and SD as fixed reference points applied to both observed and synthetic",
        },
        {
            "name": "quantile_poly_deg1",
            "description": "slope of linear fit to empirical quantile function of z-scored values, where z-scoring uses the observed dataset mean and SD as fixed reference points applied to both observed and synthetic",
        },
        {
            "name": "acf_lag1",
            "description": "lag-1 Spearman autocorrelation between consecutive time points — captures temporal persistence",
        },
        {
            "name": "spearman_trend",
            "description": "Spearman correlation between time index and population mean trajectory — captures overall trend direction",
        },
    ],
    "cross_biomarker": [
        {
            "name": "diff_corr",
            "description": "Spearman correlation of first differences between biomarker pairs — captures dynamic coupling between variables",
        }
    ],
}


class WorkspaceLogger:
    """Logs per-iteration data for one ODE sample's parameter optimization."""

    def __init__(
        self,
        problem_name: str,
        log_dir: str,
        sample_order: int,
    ) -> None:
        self._sample_dir = os.path.join(
            "workspace",
            problem_name,
            log_dir,
            f"sample_{sample_order}",
        )
        os.makedirs(self._sample_dir, exist_ok=True)
        self._iter = 0

    # ------------------------------------------------------------------
    # Public logging interface
    # ------------------------------------------------------------------

    def log_initial_inference(
        self,
        summary: Optional[str],
        params: Dict[str, Any],
        log_sl_data: Optional[Dict[str, Any]] = None,
        violation_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        d = self._next_iter_dir()
        self._write(os.path.join(d, "params.json"), {
            "iter_type": "initial_inference",
            "summary": summary or "",
            "params": params,
        })
        if log_sl_data is not None:
            self._write(os.path.join(d, "log_sl.json"), {
                "log_sl_component_descriptions": _LOG_SL_COMPONENT_DESCRIPTIONS,
                **log_sl_data,
            })
        elif violation_report is not None:
            self._write(os.path.join(d, "violation_report.json"), violation_report)

    def log_constraint_violation(
        self,
        summary: Optional[str],
        params: Dict[str, Any],
        violation_report: Dict[str, Any],
    ) -> None:
        d = self._next_iter_dir()
        self._write(os.path.join(d, "params.json"), {
            "iter_type": "constraint_violation",
            "summary": summary or "",
            "params": params,
        })
        self._write(os.path.join(d, "violation_report.json"), violation_report)

    def log_normal_iteration(
        self,
        summary: Optional[str],
        params: Dict[str, Any],
        log_sl_data: Dict[str, Any],
        failure_modes: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        d = self._next_iter_dir()
        self._write(os.path.join(d, "params.json"), {
            "iter_type": "normal",
            "summary": summary or "",
            "params": params,
        })
        self._write(os.path.join(d, "log_sl.json"), {
            "log_sl_component_descriptions": _LOG_SL_COMPONENT_DESCRIPTIONS,
            **log_sl_data,
        })
        self._write(os.path.join(d, "failure_modes.json"), failure_modes)
        self._write(os.path.join(d, "tool_results.json"), tool_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_iter_dir(self) -> str:
        d = os.path.join(self._sample_dir, f"iter_{self._iter}")
        os.makedirs(d, exist_ok=True)
        self._iter += 1
        return d

    @staticmethod
    def _write(path: str, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = ["WorkspaceLogger"]
