"""Find the best system from a log directory.

Usage:
    python find_best_system.py --log_path logs/aki_run1
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

    best_score = -float("inf")
    best_fn = None
    sample_order = None

    for file in os.listdir(log_dir):
        if file.endswith(".json"):
            with open(os.path.join(log_dir, file), "r") as f:
                sample = json.load(f)
                score = sample.get("score")
                if score is not None and score > best_score:
                    best_score = score
                    best_fn = sample["function"]
                    sample_order = sample["sample_order"]

    print("Best Score:", best_score)
    print("Sample Order:", sample_order)
    # print("\nBest Function:")
    # print(best_fn)


if __name__ == '__main__':
    main()
