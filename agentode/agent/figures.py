"""Observed vs synthetic trajectory comparison figures returned as PNG bytes.

Auto-loads observed data from `data/<problem_name>/<problem_name>.csv`.
Returns: {"mean_trajectory", "umap" (optional), "faceted_by_baseline", "diff_corr_heatmap"}.
"""

from __future__ import annotations

import io
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.lines import Line2D

from agentode.config import Config
from agentode.ode import initial_condition_utils

_DEFAULT_CONFIG = Config()


def _resolve_bin_width(bin_width: Optional[float], problem_name: Optional[str] = None) -> float:
    """Return bin_width, loading from IC config when not explicitly provided."""
    if bin_width is not None:
        return bin_width
    if problem_name is not None:
        try:
            config = initial_condition_utils.load_ic_config(problem_name)
            return initial_condition_utils.get_trajectory_bin_width(
                config, fallback=_DEFAULT_CONFIG.trajectory_bin_width
            )
        except FileNotFoundError:
            pass
    return _DEFAULT_CONFIG.trajectory_bin_width


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, lower-cased column names."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _auto_detect_vars(df: pd.DataFrame) -> List[str]:
    """Detect numeric biomarker columns (exclude id, t)."""
    exclude = {"id", "t"}
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _bin_trajectories(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """Assign each row to a fixed-width time bin."""
    t_min = float(df["t"].min())
    t_max = max(float(df["t"].max()), t_min + bin_width)
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    out = df.copy()
    out["t_bin"] = pd.cut(out["t"], bins=bins)
    return out


def _bin_stats(df_binned: pd.DataFrame, var: str) -> pd.DataFrame:
    """Per-bin mean + t-distribution 95% CI for one variable."""
    grp = df_binned.groupby("t_bin", observed=False)[var]
    st = grp.agg(["mean", "std", "count"])
    st["se"] = st["std"] / np.sqrt(st["count"])
    st["ci"] = st.apply(
        lambda r: scipy.stats.t.ppf(0.975, max(int(r["count"]) - 1, 1)) * r["se"],
        axis=1,
    )
    st["t_centre"] = [iv.left + (iv.right - iv.left) / 2 for iv in st.index]
    st = st[st["count"] > 0]
    return st


def _patient_features(df: pd.DataFrame, vars_list: Iterable[str]) -> pd.DataFrame:
    """Summarise each patient as a feature vector: mean, max, slope, std per var."""
    feats: List[Dict[str, float]] = []
    for pid, g in df.groupby("id"):
        g = g.sort_values("t")
        row: Dict[str, float] = {"id": pid}
        t_vals = g["t"].to_numpy(dtype=float)
        for v in vars_list:
            s = g[v].dropna()
            if len(s) == 0:
                continue
            vals = s.to_numpy(dtype=float)
            row[f"{v}_mean"] = float(vals.mean())
            row[f"{v}_max"] = float(vals.max())
            if len(vals) > 1:
                # Fit slope on aligned time/value pairs.
                n = len(vals)
                slope = np.polyfit(t_vals[:n], vals, 1)[0]
            else:
                slope = 0.0
            row[f"{v}_slope"] = float(slope)
            row[f"{v}_std"] = float(vals.std())
        row["n_obs"] = float(len(g))
        if len(g) > 0:
            row["duration"] = float(g["t"].max() - g["t"].min())
        else:
            row["duration"] = 0.0
        feats.append(row)
    return pd.DataFrame(feats).fillna(0.0)


def _add_baseline(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Merge each patient's first (baseline) value back onto all their rows."""
    baseline = (
        df.sort_values("t")
        .groupby("id")[var]
        .first()
        .rename("baseline")
    )
    return df.merge(baseline, on="id")


def _compute_diff_correlations(
    data: pd.DataFrame,
    vars_list: Iterable[str],
    subject_col: str = "id",
    time_col: str = "t",
) -> pd.DataFrame:
    """Spearman correlation of first differences for all variable pairs."""
    grouped = data.groupby(subject_col)
    records: List[Dict[str, float | str]] = []
    vars_list = list(vars_list)

    for i, v1 in enumerate(vars_list):
        for j, v2 in enumerate(vars_list):
            if j <= i:
                continue
            diffs1: List[float] = []
            diffs2: List[float] = []
            for _, grp in grouped:
                g = grp.sort_values(time_col)
                sub = g[[v1, v2]].dropna()
                a = sub[v1].to_numpy(dtype=float)
                b = sub[v2].to_numpy(dtype=float)
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


def _results_to_matrix(results: pd.DataFrame, vars_list: Iterable[str]) -> np.ndarray:
    """Convert long-form results DataFrame to a square symmetric matrix."""
    vars_list = list(vars_list)
    n = len(vars_list)
    mat = np.eye(n, dtype=float)
    idx = {v: i for i, v in enumerate(vars_list)}
    for _, row in results.iterrows():
        v1, v2 = [s.strip() for s in str(row["pair"]).split(" vs ")]
        i, j = idx[v1], idx[v2]
        mat[i, j] = mat[j, i] = float(row["diff_corr"])
    return mat


def _draw_heatmap(ax, matrix: np.ndarray, vars_list: Iterable[str], title: str) -> None:
    """Draw an annotated correlation heatmap on ax."""
    vars_list = list(vars_list)
    n = len(vars_list)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            v = float(matrix[i, j])
            norm_v = (v + 1.0) / 2.0
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
    ax.set_yticklabels(
        [v.replace("_", " ") for v in vars_list],
        fontsize=9,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Spearman r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _fig_to_png_bytes(fig) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes without touching disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_observed_vs_synthetic_figures(
    problem_name: str,
    synthetic_df: pd.DataFrame,
    vars: Optional[Iterable[str]] = None,
    bin_width: Optional[float] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
) -> Dict[str, bytes]:
    """Generate four comparison figures (PNG bytes) for a problem.

    Args:
        problem_name: Name of the problem, used to load observed data from
            `data/<problem_name>/<problem_name>.csv`.
        synthetic_df: Long-format synthetic data with columns `id`, `t`, and
            one or more numeric biomarker columns.
        vars: Optional iterable of biomarker columns to analyse. If None,
            numeric columns in the observed data (excluding `id`, `t`)
            are auto-detected and intersected with the synthetic columns.
        bin_width: Time bin width for mean-trajectory and stratified plots.
            Defaults to ``Config.trajectory_bin_width`` when *None*.
        umap_n_neighbors, umap_min_dist, umap_metric: UMAP hyperparameters.

    Returns:
        Dict mapping figure key -> PNG bytes:
            "mean_trajectory", "umap" (optional), "faceted_by_baseline",
            "diff_corr_heatmap".
    """
    bin_width = _resolve_bin_width(bin_width, problem_name)

    # Load observed data
    obs_path = os.path.join("data", problem_name, f"{problem_name}.csv")
    obs_df = pd.read_csv(obs_path)

    # Normalise column names
    obs = _normalize_columns(obs_df)
    synth = _normalize_columns(synthetic_df)

    # Debug: report how many synthetic trajectories (patients) are available.
    n_synth_patients = 0
    if "id" in synth.columns:
        try:
            n_synth_patients = int(synth["id"].nunique())
        except Exception:
            n_synth_patients = 0
    print(
        "[figures] synthetic trajectories for diagnosis:",
        f"n_patients={n_synth_patients}, shape={synth.shape}",
    )

    # Determine variables to analyse
    if vars is None:
        vars = _auto_detect_vars(obs)
    vars = [v for v in vars if v in obs.columns and v in synth.columns]
    if not vars:
        if n_synth_patients == 0:
            return {}
        raise ValueError(
            f"No shared numeric variables between observed {_auto_detect_vars(obs)} "
            f"and synthetic {_auto_detect_vars(synth)} ({n_synth_patients} patients)."
        )


    # Bin both datasets
    obs_b = _bin_trajectories(obs, bin_width)
    synth_b = _bin_trajectories(synth, bin_width)

    # Colours / labels (reused across figures)
    colors = {"obs": "#3266ad", "synth": "#E24B4A"}
    labels = {"obs": "Observed", "synth": "Synthetic"}
    legend_elements = [
        Line2D([0], [0], color=colors["obs"], lw=2, label="Observed"),
        Line2D(
            [0],
            [0],
            color=colors["synth"],
            lw=2,
            linestyle="--",
            label="Synthetic",
        ),
    ]

    figures: Dict[str, bytes] = {}

    # 1) Mean trajectory ± 95% CI
    n_vars = len(vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]
    fig.suptitle(
        f"Mean trajectory ± 95% CI  (bin width = {bin_width}, t-dist)",
        fontsize=13,
        y=1.02,
    )
    for ax, v in zip(axes, vars):
        for key, df_b in [("obs", obs_b), ("synth", synth_b)]:
            st = _bin_stats(df_b, v)
            t = st["t_centre"].to_numpy(dtype=float)
            mean = st["mean"].to_numpy(dtype=float)
            ci = st["ci"].to_numpy(dtype=float)
            ax.plot(
                t,
                mean,
                color=colors[key],
                lw=2,
                linestyle="-" if key == "obs" else "--",
                label=labels[key],
            )
            ax.fill_between(
                t,
                mean - ci,
                mean + ci,
                color=colors[key],
                alpha=0.15,
            )
        ax.set_xlabel("Time", fontsize=11)
        ax.set_title(v.replace("_", " ").capitalize(), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean value", fontsize=11)
    axes[-1].legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    figures["mean_trajectory"] = _fig_to_png_bytes(fig)

    # 2) UMAP embedding of per-patient summary features (optional)
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from umap import UMAP  # type: ignore

        obs_feat = _patient_features(obs, vars)
        synth_feat = _patient_features(synth, vars)
        feat_cols = [
            c for c in obs_feat.columns if c != "id" and c in synth_feat.columns
        ]

        X_obs = obs_feat[feat_cols].to_numpy(dtype=float)
        X_synth = synth_feat[feat_cols].to_numpy(dtype=float)
        X_all = np.vstack([X_obs, X_synth])
        n_obs = X_obs.shape[0]

        X_scaled = StandardScaler().fit_transform(X_all)

        emb = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
        ).fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            emb[:n_obs, 0],
            emb[:n_obs, 1],
            c=colors["obs"],
            alpha=0.55,
            s=20,
            label="Observed",
            edgecolors="none",
        )
        ax.scatter(
            emb[n_obs:, 0],
            emb[n_obs:, 1],
            c=colors["synth"],
            alpha=0.55,
            s=20,
            label="Synthetic",
            edgecolors="none",
            marker="^",
        )
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title("UMAP embedding of patient trajectory features", fontsize=12)
        ax.text(
            0.5,
            -0.1,
            f"n_neighbors={umap_n_neighbors}  |  min_dist={umap_min_dist}"
            f"  |  metric='{umap_metric}'",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
            ha="center",
        )
        ax.legend(fontsize=10, markerscale=1.4)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        figures["umap"] = _fig_to_png_bytes(fig)
    except Exception:
        # If UMAP / sklearn are not available, just skip this figure.
        pass

    # 3) Mean trajectory by baseline quartile
    q_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    fig, axes = plt.subplots(n_vars, 4, figsize=(16, 4 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Mean trajectory by baseline quartile (per variable)",
        fontsize=13,
        y=1.01,
    )
    for row, var in enumerate(vars):
        obs_bl = _add_baseline(obs, var)
        synth_bl = _add_baseline(synth, var)

        quartiles = obs_bl["baseline"].quantile(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        ).to_numpy(dtype=float)

        for qi in range(4):
            ax = axes[row][qi]
            lo, hi = quartiles[qi], quartiles[qi + 1]
            for key, df_bl, df_b in [
                ("obs", obs_bl, obs_b),
                ("synth", synth_bl, synth_b),
            ]:
                ids_q = df_bl[
                    (df_bl["baseline"] >= lo) & (df_bl["baseline"] <= hi)
                ]["id"].unique()
                sub = df_b[df_b["id"].isin(ids_q)]
                if len(sub) == 0:
                    continue
                st = _bin_stats(sub, var)
                t = st["t_centre"].to_numpy(dtype=float)
                mean = st["mean"].to_numpy(dtype=float)
                ci = st["ci"].to_numpy(dtype=float)
                ax.plot(
                    t,
                    mean,
                    color=colors[key],
                    lw=2,
                    linestyle="-" if key == "obs" else "--",
                    label=labels[key],
                )
                ax.fill_between(
                    t,
                    mean - ci,
                    mean + ci,
                    color=colors[key],
                    alpha=0.15,
                )
            if row == 0:
                ax.set_title(q_labels[qi], fontsize=10)
            if row == n_vars - 1:
                ax.set_xlabel("Time", fontsize=10)
            if qi == 0:
                ax.set_ylabel(var.replace("_", " ").capitalize(), fontsize=11)
            if row == 0 and qi == 3:
                ax.legend(handles=legend_elements, fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figures["faceted_by_baseline"] = _fig_to_png_bytes(fig)

    # 4) Difference correlation heatmap — Observed vs Synthetic
    obs_diff_res = _compute_diff_correlations(obs, vars)
    synth_diff_res = _compute_diff_correlations(synth, vars)
    obs_diff_mat = _results_to_matrix(obs_diff_res, vars)
    synth_diff_mat = _results_to_matrix(synth_diff_res, vars)

    fig_size = max(5.0, len(vars) * 1.4)
    fig, (ax_obs, ax_syn) = plt.subplots(
        1, 2, figsize=(fig_size * 2 + 1, fig_size * 0.9)
    )
    fig.suptitle(
        "Difference correlation heatmap (pairwise deletion)\n"
        "Spearman(Δv₁, Δv₂)",
        fontsize=11,
        y=1.04,
    )
    _draw_heatmap(ax_obs, obs_diff_mat, vars, "Observed")
    _draw_heatmap(ax_syn, synth_diff_mat, vars, "Synthetic")
    plt.tight_layout()
    figures["diff_corr_heatmap"] = _fig_to_png_bytes(fig)

    return figures


def plot_distribution_boxplot_comparison(
    problem_name: str,
    synthetic_df: pd.DataFrame,
    var: str,
    bin_width: Optional[float] = None,
    time_unit: str = "h",
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    dark_theme: bool = True,
) -> bytes:
    """Generate a paired box plot comparing observed vs synthetic distributions over time.

    Each time bin shows two side-by-side boxes: blue = observed, red = synthetic.
    Box spans IQR (25–75 %ile), centre bar = median, dashed whiskers = 10–90 %ile.
    Solid lines connect the medians across bins.

    Args:
        problem_name: Used to load observed data from
            ``data/<problem_name>/<problem_name>.csv``.
        synthetic_df: Long-format synthetic data with columns ``id``, ``t``, and
            at least the column named by *var*.
        var: Biomarker column to visualise (must exist in both datasets after
            column-name normalisation).
        bin_width: Width of each time bin (same units as the ``t`` column).
            Defaults to ``Config.trajectory_bin_width`` when *None*.
        time_unit: Unit suffix appended to bin labels (e.g. ``"h"`` → "0-14h").
        y_label: Y-axis label.  Defaults to the variable name.
        title: Figure title.  Defaults to
            "``<Var>`` distribution over time — observed vs synthetic".
        dark_theme: If True (default), renders with a dark background matching
            the style shown in the screenshot.

    Returns:
        PNG image as raw bytes.
    """
    import matplotlib.patches as mpatches

    bin_width = _resolve_bin_width(bin_width, problem_name)

    # ── Load & normalise ──────────────────────────────────────────────────────
    obs_path = os.path.join("data", problem_name, f"{problem_name}.csv")
    obs = _normalize_columns(pd.read_csv(obs_path))
    synth = _normalize_columns(synthetic_df)
    var = var.strip().lower()

    if var not in obs.columns:
        raise ValueError(f"Variable '{var}' not found in observed data columns: {list(obs.columns)}")
    if var not in synth.columns:
        raise ValueError(f"Variable '{var}' not found in synthetic data columns: {list(synth.columns)}")

    n_obs_patients = int(obs["id"].nunique()) if "id" in obs.columns else len(obs)
    n_syn_patients = int(synth["id"].nunique()) if "id" in synth.columns else len(synth)

    # ── Bin both datasets ─────────────────────────────────────────────────────
    obs_b = _bin_trajectories(obs, bin_width)
    synth_b = _bin_trajectories(synth, bin_width)

    # ── Compute per-bin percentile statistics ─────────────────────────────────
    pcts = (10, 25, 50, 75, 90)

    def _pct_stats(df_binned: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t_bin, grp in df_binned.groupby("t_bin", observed=False):
            vals = grp[var].dropna().to_numpy(dtype=float)
            if len(vals) < 2:
                continue
            p = np.percentile(vals, pcts)
            lo = float(t_bin.left)
            hi = float(t_bin.right)
            rows.append(
                {
                    "t_bin": t_bin,
                    "label": f"{int(lo)}-{int(hi)}{time_unit}",
                    "p10": p[0],
                    "p25": p[1],
                    "p50": p[2],
                    "p75": p[3],
                    "p90": p[4],
                    "n": len(vals),
                }
            )
        return pd.DataFrame(rows)

    obs_stats = _pct_stats(obs_b)
    syn_stats = _pct_stats(synth_b)

    # Align on shared bin labels
    all_labels = obs_stats["label"].tolist()
    obs_stats = obs_stats.set_index("label")
    syn_stats = syn_stats.set_index("label")
    shared = [lb for lb in all_labels if lb in syn_stats.index]
    if not shared:
        raise ValueError("No shared time bins between observed and synthetic data.")

    obs_stats = obs_stats.loc[shared]
    syn_stats = syn_stats.loc[shared]
    xs = np.arange(len(shared))

    # ── Colours ───────────────────────────────────────────────────────────────
    obs_color = "#4A90D9"   # blue
    syn_color = "#E24B4A"   # red/salmon
    box_half = 0.18         # half-width of each box in data units
    offset = 0.22           # centre offset from integer tick

    # ── Draw ──────────────────────────────────────────────────────────────────
    style_ctx = plt.style.context("dark_background" if dark_theme else "default")
    with style_ctx:
        fig, ax = plt.subplots(figsize=(max(10, len(shared) * 0.85), 6))

        def _draw_box(ax, x_centre, stats_row, color):
            p10, p25, p50, p75, p90 = (
                stats_row["p10"],
                stats_row["p25"],
                stats_row["p50"],
                stats_row["p75"],
                stats_row["p90"],
            )
            # IQR box
            box = mpatches.FancyBboxPatch(
                (x_centre - box_half, p25),
                2 * box_half,
                p75 - p25,
                boxstyle="square,pad=0",
                linewidth=1.5,
                edgecolor=color,
                facecolor=color + "44",  # ~27% alpha hex
            )
            ax.add_patch(box)
            # Median line
            ax.plot(
                [x_centre - box_half, x_centre + box_half],
                [p50, p50],
                color=color,
                lw=2.0,
                solid_capstyle="butt",
            )
            # Whiskers (10-90)
            for p_end, p_box in [(p10, p25), (p90, p75)]:
                ax.plot(
                    [x_centre, x_centre],
                    [p_end, p_box],
                    color=color,
                    lw=1.2,
                    linestyle="--",
                )
                ax.plot(
                    [x_centre - box_half * 0.55, x_centre + box_half * 0.55],
                    [p_end, p_end],
                    color=color,
                    lw=1.2,
                )

        for i, label in enumerate(shared):
            _draw_box(ax, xs[i] - offset, obs_stats.loc[label], obs_color)
            _draw_box(ax, xs[i] + offset, syn_stats.loc[label], syn_color)

        # Median connecting lines
        ax.plot(
            xs - offset,
            obs_stats["p50"].to_numpy(dtype=float),
            color=obs_color,
            lw=1.8,
            label="Obs median",
        )
        ax.plot(
            xs + offset,
            syn_stats["p50"].to_numpy(dtype=float),
            color=syn_color,
            lw=1.8,
            label="Syn median",
        )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_xticks(xs)
        ax.set_xticklabels(shared, fontsize=9)
        ax.set_xlim(-0.6, len(shared) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(y_label or var.replace("_", " ").capitalize(), fontsize=11)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        # ── Title and legend ──────────────────────────���───────────────────────
        ax.set_title(
            title or f"{var.replace('_', ' ').capitalize()} distribution over time"
                      " \u2014 observed vs synthetic",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        # Legend (top-right, outside plot)
        legend_patches = [
            mpatches.Patch(facecolor=obs_color + "44", edgecolor=obs_color,
                           label=f"Observed (n={n_obs_patients})"),
            mpatches.Patch(facecolor=syn_color + "44", edgecolor=syn_color,
                           label=f"Synthetic (n={n_syn_patients})"),
        ]
        legend_lines = [
            Line2D([0], [0], color=obs_color, lw=1.8, label="Obs median"),
            Line2D([0], [0], color=syn_color, lw=1.8, label="Syn median"),
            Line2D([0], [0], color="grey", lw=1.2, linestyle="--",
                   label="10\u201390% whisker"),
        ]
        ax.legend(
            handles=legend_patches + legend_lines,
            loc="upper right",
            fontsize=9,
            framealpha=0.3,
            ncol=2,
        )

        # ── Caption ───────────────────────────────────────────────────────────
        caption = (
            "Each time bin shows two paired boxes side by side: "
            "\u25a0\u00a0blue\u00a0=\u00a0observed,\u00a0"
            "\u25a0\u00a0red\u00a0=\u00a0synthetic.\u2002"
            "Box spans IQR (25\u201375%), horizontal bar = median, "
            "whiskers = 10th\u201390th percentile."
        )
        fig.text(
            0.5,
            -0.04,
            caption,
            ha="center",
            va="top",
            fontsize=8,
            color="grey",
            wrap=True,
        )

        plt.tight_layout()
        return _fig_to_png_bytes(fig)


__all__ = ["generate_observed_vs_synthetic_figures", "plot_distribution_boxplot_comparison"]
