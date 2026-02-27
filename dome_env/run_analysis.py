#!/usr/bin/env python3
"""
Process all observing nights and fit dome thermalization models.

Uses multiprocessing to parallelize across nights.

Usage:
    python dome_env/run_analysis.py
    python dome_env/run_analysis.py --max-nights 10  # quick test
    python dome_env/run_analysis.py --workers 8      # control parallelism
"""

import argparse
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dome_thermal import load_thermal_data, prepare_night, fit_all_models

RESULTS_PATH = Path(__file__).parent / "results_all_nights.csv"

SAVE_COLS = [
    "night_date", "n_points", "dome_hours", "T_floor", "delta_T_initial",
    "wind_speed_mean", "wind_speed_std",
    "m0_tau", "m0_rmse", "m0_bic",
    "m1_tau", "m1_rmse", "m1_bic",
    "m2_tau", "m2_k", "m2_dT_offset", "m2_rmse", "m2_bic",
    "m3_tau", "m3_alpha", "m3_rmse", "m3_bic",
    "m6_tau", "m6_alpha", "m6_k", "m6_rmse", "m6_bic",
]


def process_one_night(night_df_tuple, eval_hours=9.0):
    """Process a single night. Designed for multiprocessing."""
    night, night_df = night_df_tuple

    night_data = prepare_night(night_df, eval_hours=eval_hours)
    if night_data is None:
        return None

    results = fit_all_models(night_data)
    return results


def main():
    parser = argparse.ArgumentParser(description="Dome thermalization analysis")
    parser.add_argument("--max-nights", type=int, default=None)
    parser.add_argument("--eval-hours", type=float, default=9.0,
                        help="Evaluation window after dome-open (hours)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    parser.add_argument("-o", "--output", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    n_workers = args.workers or min(cpu_count(), 16)

    print("Loading thermal data...")
    df = load_thermal_data()
    nights = df["night_date"].unique()
    print(f"Found {len(nights)} nights")

    if args.max_nights:
        nights = nights[:args.max_nights]
        print(f"Limited to {len(nights)} nights")

    # Pre-split data by night for multiprocessing
    night_dfs = [(night, df[df["night_date"] == night]) for night in nights]

    print(f"Processing {len(nights)} nights with {n_workers} workers...")

    worker_fn = partial(process_one_night, eval_hours=args.eval_hours)

    with Pool(n_workers) as pool:
        results_list = pool.map(worker_fn, night_dfs)

    # Collect results
    all_results = [r for r in results_list if r is not None]
    failed = len(results_list) - len(all_results)

    # Pass 2: re-fit nights with m2_tau > 6h using a 6h window
    refit_nights = []
    refit_indices = []
    for i, r in enumerate(all_results):
        tau = r.get("m2_tau", 0)
        if tau is not None and tau > 6.0:
            night = r["night_date"]
            refit_nights.append((night, df[df["night_date"] == night]))
            refit_indices.append(i)

    if refit_nights:
        print(f"\nPass 2: re-fitting {len(refit_nights)} nights (tau > 6h) with 6h window...")
        worker_fn_6h = partial(process_one_night, eval_hours=6.0)
        with Pool(n_workers) as pool:
            refit_results = pool.map(worker_fn_6h, refit_nights)

        replaced = 0
        for idx, new_r in zip(refit_indices, refit_results):
            if new_r is not None:
                all_results[idx] = new_r
                replaced += 1
        print(f"  Replaced {replaced}/{len(refit_nights)} fits")

    # Print per-night summary
    for r in all_results:
        tau6 = r.get("m6_tau", np.nan)
        alpha6 = r.get("m6_alpha", np.nan)
        rmse6 = r.get("m6_rmse", np.nan)
        print(f"  {r['night_date']}  tau={tau6:.1f}h  alpha={alpha6:.2f}  rmse={rmse6:.3f}C")

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)
    save_cols = [c for c in SAVE_COLS if c in results_df.columns]
    results_df = results_df[save_cols]
    results_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Processed: {len(all_results)} nights ({failed} skipped)")
    print(f"Results saved to: {args.output}")

    def mad(s):
        return (s - s.median()).abs().median()

    # Summary statistics for M6 (Stratified Exchange)
    if "m6_tau" in results_df.columns:
        tau = results_df["m6_tau"].dropna()
        alpha = results_df["m6_alpha"].dropna()
        rmse = results_df["m6_rmse"].dropna()
        print(f"\nM6 (Stratified Exchange) summary:")
        print(f"  tau:   {tau.median():.1f} +/- {mad(tau):.1f} h  (median +/- MAD)")
        print(f"  alpha: {alpha.median():.2f} +/- {mad(alpha):.2f}")
        print(f"  RMSE:  {rmse.median():.3f} +/- {mad(rmse):.3f} C")

    # Model comparison
    print(f"\nModel RMSE comparison (median):")
    for m in ["m0", "m1", "m2", "m3", "m6"]:
        col = f"{m}_rmse"
        if col in results_df.columns:
            r = results_df[col].dropna()
            print(f"  {m}: {r.median():.3f} C")


if __name__ == "__main__":
    main()
