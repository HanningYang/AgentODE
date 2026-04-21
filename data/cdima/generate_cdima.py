"""
CDIMA Reaction — Synthetic Data Generator (framework-aligned)
=============================================================
Generates ID and OOD trajectory cohorts for the CDIMA system using the
same utilities the AgentODE framework uses during evaluation.

Runs both solvers side-by-side for comparison:
  - odeint: scipy LSODA, adaptive step size (accurate ground truth)
  - rk4:    torch batched RK4, fixed dt=0.4 (same as agent evaluation)

Results saved to separate subdirectories:
  data/cdima/results/odeint/
  data/cdima/results/rk4/

Parameter index convention (matches param_utils.get_default_param_distributions):
    params[0] = c4  (mean=4.0, sd=0.6)
    params[1] = a   (mean=8.9, sd=1.5)
    params[2] = b   (mean=1.4, sd=0.25)

ODE system:
    dx0/dt = -c4 * x0 * x1 / (x0^2 + 1) - x0 + a
    dx1/dt =  b * x0 * (1 - x1 / (x0^2 + 1))
"""

import copy
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.integrate import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentode.ode import initial_condition_utils
from agentode.core import param_utils
from agentode.ode.ode_simulator import (
    simulate_ode_system_torch,
    check_trajectory_normal_range_validity,
)

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIRS = {
    "odeint": Path(__file__).parent / "results" / "odeint",
    "rk4":    Path(__file__).parent / "results" / "rk4",
}
for d in OUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ── Parameter distributions (framework indexed format) ────────────────────────
PARAM_DISTRIBUTIONS_ID = {
    "0": {"mean": 4.0,  "sd": 0.6},    # c4
    "1": {"mean": 8.9,  "sd": 1.5},    # a
    "2": {"mean": 1.4,  "sd": 0.25},   # b
}

PARAM_DISTRIBUTIONS_OOD = {
    "0": {"mean": 5.5,  "sd": 0.5},    # c4
    "1": {"mean": 6.0,  "sd": 1.0},    # a
    "2": {"mean": 1.8,  "sd": 0.20},   # b
}

# OOD IC ranges
OOD_IC_RANGES = {
    "x0": {"clip_min": 3.0, "clip_max": 8.0},
    "x1": {"clip_min": 2.0, "clip_max": 6.0},
}


# ── ODE system — numpy version for odeint ─────────────────────────────────────
def system_numpy(y, t, p):
    x0, x1 = y
    denom = x0 ** 2 + 1.0
    dx0 = -p[0] * x0 * x1 / denom - x0 + p[1]
    dx1 =  p[2] * x0 * (1.0 - x1 / denom)
    return [dx0, dx1]


# ── ODE system — torch batch version for RK4 ──────────────────────────────────
def system_torch(biomarkers: torch.Tensor, params: torch.Tensor, t: float) -> torch.Tensor:
    x0 = biomarkers[..., 0]
    x1 = biomarkers[..., 1]
    denom = x0 ** 2 + 1.0
    dx0 = -params[..., 0] * x0 * x1 / denom - x0 + params[..., 1]
    dx1 =  params[..., 2] * x0 * (1.0 - x1 / denom)
    return torch.stack([dx0, dx1], dim=-1)


# ── Build OOD config ───────────────────────────────────────────────────────────
def make_ood_config(id_config: dict) -> dict:
    ood_config = copy.deepcopy(id_config)
    for biomarker, ranges in OOD_IC_RANGES.items():
        ood_config["biomarkers"][biomarker]["clip_min"] = ranges["clip_min"]
        ood_config["biomarkers"][biomarker]["clip_max"] = ranges["clip_max"]
    return ood_config


# ── Integrate with odeint ─────────────────────────────────────────────────────
def _integrate_odeint(ic_array, params, t_eval):
    n = ic_array.shape[0]
    n_timepoints = len(t_eval)
    n_biomarkers = ic_array.shape[1]
    trajectories = np.zeros((n, n_timepoints, n_biomarkers))
    for i in range(n):
        try:
            trajectories[i] = odeint(system_numpy, ic_array[i], t_eval,
                                     args=(params[i],))
        except Exception:
            trajectories[i] = np.nan
    return trajectories


# ── Integrate with torch RK4 ──────────────────────────────────────────────────
def _integrate_rk4(ic_array, params, t_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_t = simulate_ode_system_torch(
        system_func=system_torch,
        initial_conditions=ic_array,
        time_grid=t_eval,
        params=params,
        device=device,
        dtype=torch.float32,
        method="rk4",
    )
    return traj_t.detach().cpu().numpy()


# ── Simulate cohort ───────────────────────────────────────────────────────────
def simulate_cohort(config: dict, param_distributions: dict, n: int,
                    solver: str, seed: int = 42):
    assert solver in ("odeint", "rk4"), f"Unknown solver: {solver}"

    biomarker_names = initial_condition_utils.get_biomarker_order(config)
    t_eval = initial_condition_utils.get_time_grid(config)

    ic_array, _ = initial_condition_utils.generate_initial_conditions(
        config, sample_size=n, random_seed=seed,
    )
    params = param_utils.sample_params_from_distributions(
        param_distributions, n_samples=n, random_seed=seed, distribution="lognormal",
    )

    if solver == "odeint":
        trajectories = _integrate_odeint(ic_array, params, t_eval)
    else:
        trajectories = _integrate_rk4(ic_array, params, t_eval)

    valid_mask, issues = check_trajectory_normal_range_validity(
        trajectories, config=config, check_nans=True,
    )
    n_valid = int(valid_mask.sum())
    print(f"    {n_valid}/{n} valid | issues: {issues}")

    if n_valid == 0:
        print("    WARNING: no valid trajectories produced.")
        return []

    valid_traj   = trajectories[valid_mask]
    valid_ics    = ic_array[valid_mask]
    valid_params = params[valid_mask]

    results = []
    for i in range(n_valid):
        results.append({
            "t":  t_eval,
            "x0": valid_traj[i, :, 0],
            "x1": valid_traj[i, :, 1],
            "params": {
                "c4": float(valid_params[i, 0]),
                "a":  float(valid_params[i, 1]),
                "b":  float(valid_params[i, 2]),
            },
            "ic": {name: float(valid_ics[i, j]) for j, name in enumerate(biomarker_names)},
        })
    return results


# ── Save trajectories ─────────────────────────────────────────────────────────
def save_trajectories(results, path):
    rows = []
    for i, r in enumerate(results):
        for j in range(len(r["t"])):
            rows.append({
                "id": i,
                "t":  round(float(r["t"][j]),  1),
                "x0": round(float(r["x0"][j]), 6),
                "x1": round(float(r["x1"][j]), 6),
            })
    df = pd.DataFrame(rows, columns=["id", "t", "x0", "x1"])
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")


# ── Save metadata ─────────────────────────────────────────────────────────────
def save_metadata(results, path):
    meta = [
        {
            "params": {k: round(float(v), 6) for k, v in r["params"].items()},
            "ic":     {k: round(float(v), 6) for k, v in r["ic"].items()},
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    Saved: {path}")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results, title="CDIMA reaction", alpha=0.25, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=13, y=1.02)

    colors = {"x0": "#378ADD", "x1": "#D4537E"}
    all_x0 = np.array([r["x0"] for r in results])
    all_x1 = np.array([r["x1"] for r in results])
    t_grid = results[0]["t"]

    for row in all_x0:
        axes[0].plot(t_grid, row, color=colors["x0"], alpha=alpha, linewidth=0.8)
    axes[0].plot(t_grid, np.mean(all_x0, axis=0), color=colors["x0"],
                 linewidth=2.2, linestyle="--", label="Mean")
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("Concentration")
    axes[0].set_title("Activator x0(t)"); axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(True, alpha=0.2)

    for row in all_x1:
        axes[1].plot(t_grid, row, color=colors["x1"], alpha=alpha, linewidth=0.8)
    axes[1].plot(t_grid, np.mean(all_x1, axis=0), color=colors["x1"],
                 linewidth=2.2, linestyle="--", label="Mean")
    axes[1].set_xlabel("Time"); axes[1].set_ylabel("Concentration")
    axes[1].set_title("Inhibitor x1(t)"); axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(True, alpha=0.2)

    for r in results:
        axes[2].plot(r["x0"], r["x1"], color="#534AB7", alpha=alpha, linewidth=0.8)
    axes[2].set_xlabel("x0 (activator)"); axes[2].set_ylabel("x1 (inhibitor)")
    axes[2].set_title("Phase portrait")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# ── Parameter summary helper ──────────────────────────────────────────────────
def print_param_summary(results, label):
    if not results:
        print(f"  {label}: no valid trajectories.")
        return
    print(f"  {label} (n={len(results)}):")
    for key in ["c4", "a", "b"]:
        vals = [r["params"][key] for r in results]
        print(f"    {key:3s}: mean={np.mean(vals):.3f}  "
              f"sd={np.std(vals):.3f}  median={np.median(vals):.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ID  = 100
    N_OOD = 25

    id_config  = initial_condition_utils.load_ic_config("cdima")
    ood_config = make_ood_config(id_config)

    cohorts = {}   # cohorts[solver][cohort_name] = results

    for solver in ("odeint", "rk4"):
        print(f"\n{'='*50}")
        print(f"Solver: {solver}")
        print(f"{'='*50}")
        out = OUT_DIRS[solver]
        cohorts[solver] = {}

        # ID
        print("  ID cohort...")
        id_all = simulate_cohort(id_config, PARAM_DISTRIBUTIONS_ID, N_ID,
                                 solver=solver, seed=42)
        n_train = int(0.8 * len(id_all))
        cohorts[solver]["id_inference"] = id_all[:n_train]
        cohorts[solver]["id_test"]      = id_all[n_train:]
        print(f"    Split: {len(cohorts[solver]['id_inference'])} inference "
              f"/ {len(cohorts[solver]['id_test'])} ID test")

        # OOD
        print("  OOD test cohort...")
        cohorts[solver]["ood_test"] = simulate_cohort(
            ood_config, PARAM_DISTRIBUTIONS_OOD, N_OOD, solver=solver, seed=456,
        )

        # Save
        print("  Saving...")
        for name, key in [("cdima_id_inference", "id_inference"),
                           ("cdima_id_test",      "id_test"),
                           ("cdima_ood_test",     "ood_test")]:
            res = cohorts[solver][key]
            if res:
                save_trajectories(res, out / f"{name}.csv")
                save_metadata(res,     out / f"{name}_meta.json")

        # Plot
        print("  Plotting...")
        for name, key, title, alpha in [
            ("cdima_id_inference", "id_inference", f"CDIMA [{solver}] — ID inference (80%)", 0.20),
            ("cdima_id_test",      "id_test",      f"CDIMA [{solver}] — ID test (20%)",      0.30),
            ("cdima_ood_test",     "ood_test",      f"CDIMA [{solver}] — OOD test",           0.30),
        ]:
            res = cohorts[solver][key]
            if res:
                plot(res, title=title, alpha=alpha, save_path=out / f"{name}.png")

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Comparison summary")
    print(f"{'='*50}")
    for cohort_key, label in [("id_inference", "ID inference"),
                               ("id_test",      "ID test"),
                               ("ood_test",     "OOD test")]:
        print(f"\n{label}:")
        for solver in ("odeint", "rk4"):
            print_param_summary(cohorts[solver][cohort_key], solver)

    print(f"\nDone. Results saved to: {Path(__file__).parent / 'results'}")
