"""CLI for computing population-level time series summary statistics.

Usage:
    python -m analysis.ts_summary_stats --data data/aki/aki.csv
"""

import argparse

from llmode.agent.time_series_stats import save_observed_stats


def main() -> None:
    """Parse arguments and compute statistics for an observed dataset."""
    parser = argparse.ArgumentParser(
        description="Compute population-level time series summary statistics.",
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file.")
    parser.add_argument("--time_col", default="t", help="Name of time column.")
    parser.add_argument("--id_col", default="id", help="Name of identifier column.")
    parser.add_argument(
        "--agg",
        default="mean",
        choices=["mean", "median", "std", "count"],
        help="Aggregation used to build population trajectories.",
    )
    parser.add_argument(
        "--ar_order",
        default=3,
        type=int,
        help="Order of autoregressive model in model-fit statistics.",
    )
    parser.add_argument(
        "--min_patients",
        default=3,
        type=int,
        help="Minimum number of patients contributing at a time point.",
    )
    parser.add_argument(
        "--out_root",
        default="workspace/aki/stats",
        help="Root folder for saving statistics (default: workspace/<problem_name>/stats).",
    )
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Optional dataset name for the output subfolder. "
        "Defaults to empty string, which saves directly under out_root.",
    )
    args = parser.parse_args()

    csv_path, json_path = save_observed_stats(
        data_path=args.data,
        out_root=args.out_root,
        dataset_name=args.dataset_name,
        time_col=args.time_col,
        id_col=args.id_col,
        agg=args.agg,
        ar_order=args.ar_order,
        min_patients=args.min_patients,
    )

    print(f"Statistics saved to:\n  CSV : {csv_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
