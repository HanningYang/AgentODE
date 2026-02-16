"""Stub script for future CPU vs Torch backend regression tests.

This does not yet run a full torch pipeline, but is placed here to
document the intended comparison:

  1. Run a tiny log synthetic likelihood evaluation with the existing
     NumPy/Scipy backend.
  2. Run the same evaluation with a torch-based simulator / stats once
     they are implemented.
  3. Compare scores and key summary statistics for sanity.

Filling this out is left for when a first torch backend is wired into
the main evaluation path.
"""

if __name__ == "__main__":
    print(
        "CPU vs Torch regression stub.\n"
        "Once the torch backend is implemented, this script should:\n"
        "  - run CPU and torch evaluations on a tiny problem,\n"
        "  - and assert their scores and stats are close."
    )

