"""General trajectory analysis for longitudinal datasets.

Usage:
    python -m analysis.pipeline.trajectory_analysis \
        --data data/aki/aki.csv \
        --problem_name aki \
        --bin_width 24
"""

import argparse
import os
import warnings
from typing import Iterable, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.lines import Line2D


warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory figures from a longitudinal dataset.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV with columns: id, t, variables. "
        "Defaults to data/<problem_name>/<problem_name>.csv.",
    )
    parser.add_argument(
        "--problem_name",
        required=True,
        help="Problem name (used for workspace/<problem_name>/figures).",
    )
    parser.add_argument(
        "--id_col",
        default="id",
        help="Identifier column name (default: id).",
    )
    parser.add_argument(
        "--time_col",
        default="t",
        help="Time column name (default: t).",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Subset of variable columns to analyse. "
        "Default: all numeric columns except id/time.",
    )
    parser.add_argument(
        "--bin_width",
        type=float,
        required=True,
        help="Time-bin width in the same units as the time column.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for figures. "
        "Defaults to workspace/<problem_name>/figures.",
    )
    return parser.parse_args()


def bin_trajectories(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    t_max = max(df["t"].max(), bin_width)
    bins = np.arange(0, t_max + bin_width, bin_width)
    df_b = df.copy()
    df_b["t_bin"] = pd.cut(df_b["t"], bins=bins)
    return df_b


def bin_stats(df_binned: pd.DataFrame, var: str) -> pd.DataFrame:
    grp = df_binned.groupby("t_bin", observed=False)[var]
    st = grp.agg(["mean", "std", "count"])
    st["se"] = st["std"] / np.sqrt(st["count"])
    st["ci"] = st.apply(
        lambda r: scipy.stats.t.ppf(0.975, max(int(r["count"]) - 1, 1)) * r["se"],
        axis=1,
    )
    st["t_centre"] = [iv.left + (iv.right - iv.left) / 2 for iv in st.index]
    return st[st["count"] > 0]


def add_baseline(df: pd.DataFrame, var: str) -> pd.DataFrame:
    baseline = (
        df.sort_values("t")
        .groupby("id")[var]
        .first()
        .rename("baseline")
    )
    return df.merge(baseline, on="id")


def compute_diff_correlations(
    df: pd.DataFrame,
    vars_list: Iterable[str],
    subject_col: str = "id",
    time_col: str = "t",
) -> pd.DataFrame:
    grouped = df.groupby(subject_col)
    records: List[dict] = []

    vars_list = list(vars_list)
    for i, v1 in enumerate(vars_list):
        for j, v2 in enumerate(vars_list):
            if j <= i:
                continue

            diffs1, diffs2 = [], []
            for _, grp in grouped:
                g = grp.sort_values(time_col)
                sub = g[[v1, v2]].dropna()
                a = sub[v1].to_numpy()
                b = sub[v2].to_numpy()
                if min(len(a), len(b)) > 1:
                    diffs1.extend(np.diff(a).tolist())
                    diffs2.extend(np.diff(b).tolist())

            n = len(diffs1)
            if n > 1:
                dc, _ = scipy.stats.spearmanr(diffs1, diffs2)
                dc = 0.0 if np.isnan(dc) else float(dc)
            else:
                dc = 0.0

            records.append(
                {
                    "pair": f"{v1} vs {v2}",
                    "diff_corr": round(dc, 6),
                    "n_diff_pairs": n,
                }
            )

    return pd.DataFrame(records)


def results_to_matrix(
    results: pd.DataFrame, vars_list: Iterable[str], col: str
) -> np.ndarray:
    vars_list = list(vars_list)
    n = len(vars_list)
    mat = np.eye(n)
    idx = {v: i for i, v in enumerate(vars_list)}
    for _, row in results.iterrows():
        v1, v2 = [s.strip() for s in str(row["pair"]).split(" vs ")]
        i, j = idx[v1], idx[v2]
        mat[i, j] = mat[j, i] = float(row[col])
    return mat


def draw_heatmap(ax, matrix: np.ndarray, vars_list: Iterable[str], title: str) -> None:
    vars_list = list(vars_list)
    n = len(vars_list)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            norm_v = (v + 1) / 2
            bg = plt.cm.RdBu_r(norm_v)
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=txt_color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [v.replace("_", "\n") for v in vars_list],
        fontsize=9,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels([v.replace("_", " ") for v in vars_list], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Spearman r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or os.path.join(
        "workspace", args.problem_name, "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    data_path = args.data or os.path.join("data", args.problem_name, f"{args.problem_name}.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    id_col = args.id_col.strip().lower()
    time_col = args.time_col.strip().lower()
    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Expected id_col='{id_col}' and time_col='{time_col}' in data.")

    df = df.rename(columns={id_col: "id", time_col: "t"})

    if args.vars is None:
        exclude = {"id", "t"}
        vars_list = [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        vars_list = [v.strip().lower() for v in args.vars]
    vars_list = [v for v in vars_list if v in df.columns]
    if not vars_list:
        raise ValueError("No variable columns selected.")

    n_vars = len(vars_list)
    print(f"Patients  : {df['id'].nunique()}")
    print(f"Rows      : {len(df):,}")
    print(f"Variables : {vars_list}")

    df_b = bin_trajectories(df, bin_width=float(args.bin_width))

    # Fig 1: mean ± 95% CI
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]

    fig.suptitle(
        f"Mean trajectory ± 95% CI  (bin width={args.bin_width}, t-dist)",
        fontsize=13,
        y=1.02,
    )

    color = "#3266ad"
    legend_handles = [
        Line2D([0], [0], color=color, lw=2, label="Mean"),
        mpatches.Patch(facecolor=color, alpha=0.3, label="95% CI"),
    ]

    for ax, v in zip(axes, vars_list):
        st = bin_stats(df_b, v)
        t_vals = st["t_centre"].values
        ax.plot(t_vals, st["mean"].values, color=color, lw=2, label="Mean")
        ax.fill_between(
            t_vals,
            (st["mean"] - st["ci"]).values,
            (st["mean"] + st["ci"]).values,
            color=color,
            alpha=0.15,
            label="95% CI",
        )
        ax.set_xlabel("Time", fontsize=11)
        ax.set_title(v.replace("_", " ").capitalize(), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean value", fontsize=11)
    axes[-1].legend(handles=legend_handles, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mean_trajectory.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Fig 2: mean trajectory by baseline quartile
    q_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    q_colors = ["#378ADD", "#1D9E75", "#BA7517", "#A32D2D"]

    fig, axes = plt.subplots(n_vars, 4, figsize=(16, 4 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Mean trajectory ± 95% CI by baseline quartile",
        fontsize=13,
        y=1.01,
    )

    for row, var in enumerate(vars_list):
        df_bl = add_baseline(df, var)
        quartiles = df_bl["baseline"].quantile(
            [0, 0.25, 0.5, 0.75, 1.0]
        ).values

        for qi in range(4):
            ax = axes[row][qi]
            col = q_colors[qi]
            lo = quartiles[qi]
            hi = quartiles[qi + 1]

            ids_q = df_bl[
                (df_bl["baseline"] >= lo) & (df_bl["baseline"] <= hi)
            ]["id"].unique()
            sub = df_b[df_b["id"].isin(ids_q)]

            if len(sub) == 0:
                ax.set_visible(False)
                continue

            st = bin_stats(sub, var)
            t_vals = st["t_centre"].values
            ax.plot(t_vals, st["mean"].values, color=col, lw=2)
            ax.fill_between(
                t_vals,
                (st["mean"] - st["ci"]).values,
                (st["mean"] + st["ci"]).values,
                color=col,
                alpha=0.15,
            )

            ax.text(
                0.97,
                0.97,
                f"n={len(ids_q)}",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="top",
                color="gray",
            )

            if row == 0:
                ax.set_title(q_labels[qi], fontsize=10)
            if row == n_vars - 1:
                ax.set_xlabel("Time", fontsize=10)
            if qi == 0:
                ax.set_ylabel(var.replace("_", " ").capitalize(), fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "faceted_by_baseline.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Fig 3: difference correlation heatmap
    diff_results = compute_diff_correlations(df, vars_list)
    diff_mat = results_to_matrix(diff_results, vars_list, "diff_corr")

    fig_size = max(5, n_vars * 1.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    fig.suptitle(
        "Difference correlation heatmap (pairwise deletion)\nSpearman(Δv₁, Δv₂)",
        fontsize=11,
        y=1.04,
    )

    draw_heatmap(ax, diff_mat, vars_list, "Difference correlation")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "diff_corr_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    print("\nDifference correlations (pairwise deletion):")
    print(diff_results.to_string(index=False))
    print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
