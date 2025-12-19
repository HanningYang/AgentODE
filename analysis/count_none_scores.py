"""Count how many sampled functions have `None` as their score.

Usage:
    python analysis/count_none_scores.py --log_path logs/aki_run3
"""

import argparse
import json
import os
from typing import List, Tuple


def collect_samples(log_path: str) -> List[Tuple[int, float | None]]:
    """Return a list of (sample_order, score) from a log directory."""
    samples_dir = os.path.join(log_path, "samples")
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"'samples' directory not found under {log_path}")

    results: List[Tuple[int, float | None]] = []
    for fname in os.listdir(samples_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(samples_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        sample_order = data.get("sample_order")
        score = data.get("score")
        if sample_order is None:
            # Try to infer from filename like samples_123.json
            try:
                base = os.path.splitext(fname)[0]
                sample_order = int(base.split("_")[-1])
            except Exception:
                sample_order = -1
        results.append((int(sample_order), score))

    results.sort(key=lambda x: x[0])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Count how many sampled functions have None as score."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to a log directory (e.g. logs/aki_run3)",
    )
    args = parser.parse_args()

    samples = collect_samples(args.log_path)
    if not samples:
        print(f"No sample JSON files found under {args.log_path}/samples")
        return

    total = len(samples)
    none_scores = [(order, s) for order, s in samples if s is None]
    n_none = len(none_scores)
    n_scored = total - n_none

    print(f"Log directory : {args.log_path}")
    print(f"Total samples : {total}")
    print(f"Scored        : {n_scored}")
    print(f"None scores   : {n_none}")
    if total > 0:
        frac_none = n_none / total
        print(f"Fraction None : {frac_none:.3f}")

    if n_none > 0:
        print("\nSample orders with None score:")
        print(", ".join(str(order) for order, _ in none_scores))


if __name__ == "__main__":
    main()

