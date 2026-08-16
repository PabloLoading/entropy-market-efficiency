"""Run E1 → E2 → E3 → Sensitivity sequentially and write outputs/results.json."""
from __future__ import annotations

import time

from helpers import RESULTS_PATH

if RESULTS_PATH.exists():
    RESULTS_PATH.unlink()

import run_e1_static
import run_e2_predictive
import run_e3_indices
import run_e4_subsets
import run_e5_cross
import run_sensitivity


def main():
    for label, fn in [
        ("E1 static", run_e1_static.main),
        ("E2 predictive", run_e2_predictive.main),
        ("E3 indices", run_e3_indices.main),
        ("E4 subsets", run_e4_subsets.main),
        ("E5 cross-product", run_e5_cross.main),
        ("Sensitivity", run_sensitivity.main),
    ]:
        t0 = time.perf_counter()
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        fn()
        print(f"  ({time.perf_counter() - t0:.1f}s)")
    print(f"\nDone. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
