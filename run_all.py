#!/usr/bin/env python
"""Reproduce every result in the repository with a single command.

Usage:
    python run_all.py          # full run (tests + training + sweep)
    python run_all.py --quick  # tests only (< 10 seconds)
"""

import subprocess
import sys
import time


def run(label, cmd):
    """Run a command, print its output, and abort on failure."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED ({elapsed:.1f}s)")
        sys.exit(1)
    print(f"\n  OK ({elapsed:.1f}s)")


if __name__ == "__main__":
    quick = "--quick" in sys.argv

    run("Part 1: Gradient checks",
        "python -m tests.test_gradients")

    run("Part 1: Masked-weight correctness",
        "python -m tests.test_masked")

    if not quick:
        run("Part 2: Basic MLP training",
            "python -m train.train_basic")

        run("Part 3: Self-pruning training (saliency, 90%)",
            "python -m train.train_prune")

        run("Part 3: Self-pruning with regrowth",
            "python -m train.train_prune --regrowth --sparsity 0.9")

        run("Part 4: Pareto sweep (50 runs)",
            "python -m train.pareto_sweep")

    print(f"\n{'='*70}")
    print(f"  ALL {'TESTS' if quick else 'STEPS'} PASSED")
    print(f"{'='*70}")
