"""Find the best system from a log directory.

Usage:
    python analysis/find_best_system.py --log_path logs/aki_run1
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Find the best system from log directory')
    parser.add_argument('--log_path', type=str, required=True,
                       help='Path to the log directory')
    args = parser.parse_args()

    log_dir = os.path.join(args.log_path, 'samples')

    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} does not exist")
        return

    scored_samples = []

    for file in os.listdir(log_dir):
        if file.endswith(".json"):
            with open(os.path.join(log_dir, file), "r") as f:
                sample = json.load(f)
                score = sample.get("score")
                if score is not None:
                    scored_samples.append({
                        "score": score,
                        "function": sample.get("function"),
                        "sample_order": sample.get("sample_order"),
                        "param_distributions": sample.get("param_distributions"),
                        "filename": file,
                    })

    if not scored_samples:
        print("No scored systems found in the log directory.")
        return

    scored_samples.sort(key=lambda s: s["score"], reverse=True)
    top_k = min(10, len(scored_samples))

    print(f"Found {len(scored_samples)} scored systems. Showing top {top_k}:")
    for rank, sample in enumerate(scored_samples[:top_k], start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {sample['score']}")
        print(f"Sample Order: {sample.get('sample_order')}")
        # print(f"Param Distributions: {sample.get('param_distributions')}")
        print(f"File: {sample.get('filename')}")
        # print("\nFunction:")
        # print(sample.get("function"))
    print("=" * 80)


if __name__ == '__main__':
    main()
