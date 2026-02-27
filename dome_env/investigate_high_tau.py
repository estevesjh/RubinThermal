#!/usr/bin/env python3
"""
Investigate nights with high thermalization tau (> 4 hours).

For each high-tau night, generates a diagnostic plot showing:
- T_in and T_out from -1h to sunrise (sunAlt = -15)
- M2 fit overlay
- delta_T evolution
- dT_out/dt (cooling rate of outside air)
- Wind speed
- Dome open status

Also classifies nights into physical groups based on their
temperature/wind signatures.

Usage:
    python dome_env/investigate_high_tau.py
    python dome_env/investigate_high_tau.py --tau-threshold 6.0
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dome_thermal import (
    load_thermal_data, prepare_night, smooth_temperatures,
    fit_model2, DATA_PATH,
)

FIGURES_DIR = Path(__file__).parent / "figures" / "high_tau"
RESULTS_PATH = Path(__file__).parent / "results_all_nights.csv"


def load_full_night(df, night_date):
    """Load full night data (not just the fit window)."""
    night_df = df[df["night_date"] == night_date].copy()
    if night_df.empty:
        return None
    return night_df


def classify_night(night_df, nd, fit_result):
    """
    Classify a high-tau night into physical groups.

    Groups:
    - 'small_deltaT': Initial |delta_T| < 0.5C — no cooling signal
    - 'rewarming': Inside temp rises later in night (dome air warms)
    - 'rapid_cooling': Outside drops > 3C/h at some point
    - 'calm': Mean wind < 2 m/s
    - 'intermittent_dome': Dome opens/closes multiple times
    - 'normal_slow': None of the above — just slow equilibration
    """
    tags = []

    T_in = nd["T_in"]
    T_out = nd["T_out"]
    t = nd["t"]
    valid = np.isfinite(T_in) & np.isfinite(T_out)

    if valid.sum() < 30:
        return ["insufficient_data"]

    tv = t[valid]
    Ti = T_in[valid]
    To = T_out[valid]

    # 1. Small initial delta_T
    delta_T0 = abs(Ti[0] - To[0])
    if delta_T0 < 0.5:
        tags.append("small_deltaT")

    # 2. Rewarming: inside temp rises by > 0.3C at any point after t=1h
    post_1h = tv > 1.0
    if post_1h.any():
        Ti_post = Ti[post_1h]
        # Check if T_in increases by more than 0.3C from its minimum
        min_after_1h = Ti_post.min() if len(Ti_post) > 0 else Ti[0]
        max_after_min = Ti_post[np.argmin(Ti_post):].max() if len(Ti_post) > 1 else min_after_1h
        if max_after_min - min_after_1h > 0.3:
            tags.append("rewarming")

    # 3. Rapid outside cooling: dT_out/dt < -3 C/h at any point
    if len(To) > 10:
        dTout_dt = np.gradient(To, tv)  # C/h
        if np.min(dTout_dt) < -3.0:
            tags.append("rapid_outside_cooling")

    # 4. Calm wind
    wind = nd.get("wind_speed")
    if wind is not None:
        mean_wind = np.nanmean(wind)
        if mean_wind < 2.0:
            tags.append("calm")

    # 5. Intermittent dome
    if "dome_open" in night_df.columns:
        dome_open = night_df["dome_open"].fillna(False).values
        # Count transitions from open to closed
        transitions = np.diff(dome_open.astype(int))
        n_closings = np.sum(transitions == -1)
        if n_closings >= 2:
            tags.append("intermittent_dome")

    if not tags:
        tags.append("normal_slow")

    return tags


def plot_single_night(night_date, df, results_row):
    """Generate a diagnostic multi-panel plot for one high-tau night."""
    night_df = load_full_night(df, night_date)
    if night_df is None:
        return None

    nd = prepare_night(night_df, eval_hours=9.0)
    if nd is None:
        return None

    # Full night data for context (not just fit window)
    T_in_raw = night_df["inside_m2_temp_mean"].values
    T_out_raw = night_df["outside_temp_mean"].values
    T_in_sm = smooth_temperatures(T_in_raw, window_min=15)
    T_out_sm = smooth_temperatures(T_out_raw, window_min=61)

    t_open = nd["t_open"]
    t_full = (night_df.index - t_open).total_seconds() / 3600.0

    # Sun altitude for x-axis limit
    sun_alt = None
    if "sunAltitude" in night_df.columns:
        sun_alt = night_df["sunAltitude"].values

    # Find sunrise (sunAlt crossing -15 from below, after t_open)
    t_end_plot = 9.0  # default
    if sun_alt is not None:
        post_open = t_full > 0
        if post_open.any():
            sun_post = sun_alt[post_open]
            t_post = t_full[post_open]
            # Find where sun altitude crosses -15 going up
            for j in range(1, len(sun_post)):
                if np.isfinite(sun_post[j]) and np.isfinite(sun_post[j-1]):
                    if sun_post[j-1] < -15 and sun_post[j] >= -15:
                        t_end_plot = t_post[j]
                        break

    # Fit M2 on the prepared window
    valid = np.isfinite(nd["T_in"]) & np.isfinite(nd["T_out"])
    tv = nd["t"][valid]
    Ti = nd["T_in"][valid]
    To = nd["T_out"][valid]
    r2 = fit_model2(tv, Ti, To)

    # Classify
    tags = classify_night(night_df, nd, r2)

    # --- Plot ---
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.2, 1.2, 1], hspace=0.08)

    xlim = (-1.0, max(t_end_plot + 0.5, 6.0))

    # Panel 1: Temperatures + fit
    ax1 = fig.add_subplot(gs[0])
    mask_plot = (t_full >= xlim[0]) & (t_full <= xlim[1])
    ax1.plot(t_full[mask_plot], T_in_raw[mask_plot], color="lightcoral",
             alpha=0.3, linewidth=0.5)
    ax1.plot(t_full[mask_plot], T_in_sm[mask_plot], color="tab:red",
             linewidth=2, label="$T_{in}$ (smoothed)")
    ax1.plot(t_full[mask_plot], T_out_sm[mask_plot], color="tab:blue",
             linewidth=2, label="$T_{out}$ (smoothed)")
    ax1.plot(tv, r2["pred"], "k--", linewidth=2,
             label=f"M2: $\\tau$={r2['tau']:.1f}h, k={r2['k']:.2f}")
    ax1.axhline(nd["T_floor"], color="gray", linewidth=1, linestyle=":",
                alpha=0.5, label=f"$T_{{floor}}$={nd['T_floor']:.1f}C")
    ax1.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

    tau_str = f"$\\tau$ = {r2['tau']:.1f} h"
    tag_str = ", ".join(tags)
    ax1.set_title(f"{night_date}  |  {tau_str}  |  RMSE = {r2['rmse']:.3f}C  |  [{tag_str}]",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Temperature (C)")
    ax1.legend(fontsize=9, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)

    # Panel 2: Delta T
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    delta_T = T_in_sm - T_out_sm
    ax2.plot(t_full[mask_plot], delta_T[mask_plot], color="purple", linewidth=1.5)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
    ax2.set_ylabel("$\\Delta T$ (C)")
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: dT_out/dt
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    dt_hours = np.gradient(t_full)
    dt_hours[dt_hours == 0] = 1.0 / 60.0
    dTout = np.gradient(T_out_sm) / dt_hours  # C/h
    dTout_clipped = np.clip(dTout, -5, 5)
    ax3.plot(t_full[mask_plot], dTout_clipped[mask_plot], color="tab:orange",
             linewidth=1)
    ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
    ax3.set_ylabel("$dT_{out}/dt$ (C/h)")
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # Panel 4: Wind speed + dome open
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if "wind_speed_mean" in night_df.columns:
        ws = night_df["wind_speed_mean"].values
        ax4.plot(t_full[mask_plot], ws[mask_plot], color="teal", linewidth=1,
                 label="Wind speed")
        ax4.set_ylabel("Wind (m/s)")

    # Shade dome-open periods
    if "dome_open" in night_df.columns:
        dome = night_df["dome_open"].fillna(False).values.astype(float)
        ax4.fill_between(t_full[mask_plot], 0, dome[mask_plot] * ax4.get_ylim()[1],
                         color="green", alpha=0.1, label="Dome open")

    ax4.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
    ax4.set_xlabel("Hours from dome open")
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=8, loc="upper right")

    plt.savefig(FIGURES_DIR / f"high_tau_{night_date}.png", bbox_inches="tight")
    plt.close()

    return {
        "night_date": night_date,
        "m2_tau": r2["tau"],
        "m2_k": r2["k"],
        "m2_rmse": r2["rmse"],
        "delta_T_initial": Ti[0] - To[0],
        "tags": tags,
        "wind_mean": np.nanmean(nd["wind_speed"]) if nd["wind_speed"] is not None else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Investigate high-tau nights")
    parser.add_argument("--tau-threshold", type=float, default=4.0,
                        help="Minimum tau to investigate (hours)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_thermal_data()
    results = pd.read_csv(RESULTS_PATH)

    # Filter to high-tau nights
    high_tau = results[
        (results["m2_tau"] > args.tau_threshold) &
        (results["m2_tau"] < 23.0) &  # exclude bound hits
        (results["m2_rmse"] < 1.5)
    ].copy()

    print(f"Found {len(high_tau)} nights with tau > {args.tau_threshold}h")
    print(f"  tau range: {high_tau['m2_tau'].min():.1f} - {high_tau['m2_tau'].max():.1f} h")

    # Generate plots and classify
    all_info = []
    for i, (_, row) in enumerate(high_tau.iterrows()):
        night = row["night_date"]
        print(f"[{i+1}/{len(high_tau)}] {night} (tau={row['m2_tau']:.1f}h)  ", end="", flush=True)

        info = plot_single_night(night, df, row)
        if info is not None:
            all_info.append(info)
            print(f"tags: {info['tags']}")
        else:
            print("skip")

    # Summary by group
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION SUMMARY ({len(all_info)} nights)")
    print(f"{'='*70}")

    # Count tags
    from collections import Counter
    tag_counter = Counter()
    for info in all_info:
        for tag in info["tags"]:
            tag_counter[tag] += 1

    print("\nTag frequency:")
    for tag, count in tag_counter.most_common():
        pct = 100 * count / len(all_info)
        print(f"  {tag:25s}: {count:3d} ({pct:.0f}%)")

    # Stats per group
    print("\nPer-group tau statistics:")
    for tag in tag_counter:
        nights_with_tag = [i for i in all_info if tag in i["tags"]]
        taus = [i["m2_tau"] for i in nights_with_tag]
        winds = [i["wind_mean"] for i in nights_with_tag if np.isfinite(i["wind_mean"])]
        print(f"\n  {tag} ({len(nights_with_tag)} nights):")
        print(f"    tau:  median={np.median(taus):.1f}h, "
              f"mean={np.mean(taus):.1f}h, range=[{np.min(taus):.1f}, {np.max(taus):.1f}]")
        if winds:
            print(f"    wind: median={np.median(winds):.1f}m/s, "
                  f"mean={np.mean(winds):.1f}m/s")

    # Save classification
    info_df = pd.DataFrame([{
        "night_date": i["night_date"],
        "m2_tau": i["m2_tau"],
        "m2_k": i["m2_k"],
        "m2_rmse": i["m2_rmse"],
        "delta_T_initial": i["delta_T_initial"],
        "wind_mean": i["wind_mean"],
        "tags": "|".join(i["tags"]),
    } for i in all_info])
    out_path = Path(__file__).parent / "high_tau_classification.csv"
    info_df.to_csv(out_path, index=False)
    print(f"\nClassification saved to: {out_path}")
    print(f"Plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
