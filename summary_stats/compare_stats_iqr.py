"""Compare summary statistics between observed and synthetic datasets using IQR normalisation.

This is a CLI wrapper around the core MNTD implementation in
`llmode.metrics.mntd_score`.

Usage:
    python summary_stats/compare_stats_iqr.py --obs data/real.csv --syn data/synthetic.csv
    python summary_stats/compare_stats_iqr.py --obs data/real.csv --syn data/synthetic.csv \
        --out summary_stats/comparison_iqr.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from llmode.metrics.mntd_score import compare_stats_iqr_from_dfs


def run(obs_path: str, syn_path: str, out_path: str) -> None:
    """Load CSVs, run IQR-normalised comparison, and save to disk."""
    print(f"\nLoading observed:  {obs_path}")
    obs_df = pd.read_csv(obs_path)
    print(f"  Shape: {obs_df.shape}  |  Trajectories: {obs_df['id'].nunique()}")

    print(f"\nLoading synthetic: {syn_path}")
    syn_df = pd.read_csv(syn_path)
    print(f"  Shape: {syn_df.shape}  |  Trajectories: {syn_df['id'].nunique()}")

    result, agg_score = compare_stats_iqr_from_dfs(obs_df, syn_df, out_path=out_path)

    print(f"\nSaved: {out_path}")
    print(f"  Rows:              {len(result)}")
    n_nan = result["normalized_diff"].isna().sum()
    print(f"  NaN norm diffs:    {n_nan} ({100*n_nan/len(result):.1f}%)")
    print(f"\nPreview (first 10 rows):")
    print(result.head(10).to_string(index=False))
    print(f"\n{'='*60}")
    print(f"  Aggregate Score (Mean Normalized Trajectory Discrepancy): {agg_score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare observed vs. synthetic statistics using IQR normalization.",
    )
    parser.add_argument("--obs", required=True, help="Observed data CSV (columns: id, t, ...)")
    parser.add_argument("--syn", required=True, help="Synthetic data CSV (columns: id, t, ...)")
    parser.add_argument(
        "--out",
        default="summary_stats/comparison_iqr.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    run(obs_path=args.obs, syn_path=args.syn, out_path=args.out)

