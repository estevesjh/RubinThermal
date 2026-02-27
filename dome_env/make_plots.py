#!/usr/bin/env python3
"""
Generate figures for the dome thermal environment report.

Focused on tau characterization from the winning model (M2: derivative-coupled).

Usage:
    python dome_env/make_plots.py
    python dome_env/make_plots.py --example-night 2025-11-15
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dome_thermal import (
    load_thermal_data, prepare_night,
    fit_model0, fit_model1, fit_model2, fit_model3, fit_model6,
)

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_PATH = Path(__file__).parent / "results_all_nights.csv"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "figure.dpi": 150,
})


def load_results():
    return pd.read_csv(RESULTS_PATH)


def clean_results(results, tau_col="m6_tau", rmse_col="m6_rmse"):
    """Filter out failed fits (bound-hitting, high RMSE)."""
    r = results.copy()
    r = r[r[tau_col].notna()]
    r = r[r[tau_col] < 23.0]       # remove upper-bound hits
    r = r[r[tau_col] > 0.15]       # remove lower-bound hits
    r = r[r[rmse_col] < 1.0]       # remove bad fits
    return r


def plot_tau_distribution(results):
    """Fig 1: Tau histogram for M6 (ground-coupled)."""
    r = clean_results(results)
    tau = r["m6_tau"]
    med = tau.median()
    mad = (tau - med).abs().median()
    q25, q75 = tau.quantile(0.25), tau.quantile(0.75)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(tau, bins=25, color="steelblue", edgecolor="white",
                                alpha=0.85, zorder=2)

    # Median + IQR
    ax.axvline(med, color="firebrick", linewidth=2.5, linestyle="-",
               label=f"Median: {med:.1f} h", zorder=3)
    ax.axvspan(q25, q75, color="firebrick", alpha=0.08,
               label=f"IQR: [{q25:.1f}, {q75:.1f}] h", zorder=1)

    ax.set_xlabel(r"Time Constant $\tau$ (hours)")
    ax.set_ylabel("Number of Nights")
    ax.set_title(r"Dome Thermalization $\tau$ — Ground-Coupled Model (M6)")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2, zorder=0)

    # Annotate stats
    stats = (f"N = {len(tau)} nights\n"
             f"Median = {med:.2f} h\n"
             f"MAD = {mad:.2f} h\n"
             f"Mean = {tau.mean():.2f} h\n"
             f"Std = {tau.std():.2f} h")
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, fontsize=11,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tau_distribution.png", bbox_inches="tight")
    print(f"Saved: tau_distribution.png")
    plt.close()


def plot_tau_by_month(results):
    """Fig 2: Tau by month — box plot + table printed."""
    r = clean_results(results)
    r["month"] = pd.to_datetime(r["night_date"]).dt.to_period("M")
    r = r.sort_values("month")

    months = r["month"].unique()
    month_labels = [str(m) for m in months]
    month_data = [r[r["month"] == m]["m6_tau"].values for m in months]

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(month_data, tick_labels=month_labels, patch_artist=True,
                    widths=0.55, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))

    # Color boxes by season
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, data in enumerate(month_data):
        jitter = np.random.normal(0, 0.08, size=len(data))
        ax.scatter(np.full_like(data, i + 1) + jitter, data,
                   color="navy", s=15, alpha=0.5, zorder=3)

    ax.set_xlabel("Month")
    ax.set_ylabel(r"$\tau$ (hours)")
    ax.set_title(r"Dome Thermalization $\tau$ by Month (M6: Ground-Coupled)")
    ax.grid(True, alpha=0.2, axis="y")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tau_by_month.png", bbox_inches="tight")
    print(f"Saved: tau_by_month.png")
    plt.close()

    # Print table
    print("\n  Month        | N  | Median | Mean  | Std   | Min   | Max")
    print("  " + "-" * 65)
    for m, data in zip(month_labels, month_data):
        if len(data) > 0:
            print(f"  {m:12s} | {len(data):2d} | {np.median(data):5.2f}  | "
                  f"{np.mean(data):5.2f} | {np.std(data):5.2f} | "
                  f"{np.min(data):5.2f} | {np.max(data):5.2f}")


def plot_model_comparison(results):
    """Fig 3: RMSE box plot across all models."""
    r = results.dropna(subset=["m2_rmse"])  # keep all nights for fair comparison
    r = r[r["m2_rmse"] < 2.0]  # only remove extreme outliers

    rmse_data = []
    labels = []
    colors_list = []
    model_info = [
        ("m0", "M0\nExponential", "#bbb"),
        ("m1", "M1\nBasic ODE", "#8cb4d4"),
        ("m2", "M2\nForced\nVentilation", "#6baed6"),
        ("m3", "M3\nStratified", "#9ecae1"),
        ("m6", "M6\nStratified\n+ Wind", "#2171b5"),
    ]

    for m, label, c in model_info:
        col = f"{m}_rmse"
        if col in r.columns:
            vals = r[col].dropna()
            vals = vals[vals < 2.0]
            rmse_data.append(vals.values)
            labels.append(label)
            colors_list.append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(rmse_data, tick_labels=labels, patch_artist=True,
                    widths=0.5, showfliers=False)

    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    for i, d in enumerate(rmse_data):
        med = np.median(d)
        ax.text(i + 1, med - 0.025, f"{med:.3f}", ha="center", fontsize=10,
                fontweight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.7, pad=2, boxstyle="round,pad=0.2"))

    ax.set_ylabel("RMSE (C)")
    ax.set_title("Model Comparison: Fit Quality Across All Nights")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", bbox_inches="tight")
    print(f"Saved: model_comparison.png")
    plt.close()


def plot_tau_vs_wind(results):
    """Fig 4: Tau vs wind speed scatter with regression."""
    r = clean_results(results)
    r = r.dropna(subset=["wind_speed_mean"])

    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(r["wind_speed_mean"], r["m6_tau"],
                    c=r["m6_alpha"], cmap="viridis", vmin=0, vmax=1,
                    s=40, alpha=0.7, edgecolors="gray", linewidth=0.3, zorder=3)
    cb = plt.colorbar(sc, ax=ax, label=r"$\alpha$ (air fraction)", pad=0.02)

    # Regression line
    z = np.polyfit(r["wind_speed_mean"], r["m6_tau"], 1)
    corr = r["wind_speed_mean"].corr(r["m6_tau"])
    x_fit = np.linspace(r["wind_speed_mean"].min(), r["wind_speed_mean"].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), "k--", linewidth=2, alpha=0.6,
            label=f"r = {corr:.2f}")

    ax.set_xlabel("Mean Wind Speed (m/s)")
    ax.set_ylabel(r"$\tau$ (hours)")
    ax.set_title(r"Thermalization $\tau$ vs Wind Speed (M6: Ground-Coupled)")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tau_vs_wind.png", bbox_inches="tight")
    print(f"Saved: tau_vs_wind.png")
    plt.close()


def plot_rmse_vs_wind(results):
    """Fig 5: RMSE vs wind speed for M2 vs M4, testing buoyancy hypothesis."""
    r = results.dropna(subset=["m2_rmse", "m6_rmse", "wind_speed_mean"])
    r = r[r["m2_rmse"] < 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(r["wind_speed_mean"], r["m2_rmse"], s=30, alpha=0.6,
               color="#6baed6", label="M2 (Derivative)", zorder=3)
    ax.scatter(r["wind_speed_mean"], r["m6_rmse"], s=30, alpha=0.6,
               color="#2171b5", label="M6 (Ground-Coupled)", zorder=3)

    # Regression lines
    for col, c, label in [("m2_rmse", "#6baed6", "M2"),
                           ("m6_rmse", "#2171b5", "M6")]:
        valid = r[[col, "wind_speed_mean"]].dropna()
        corr = valid[col].corr(valid["wind_speed_mean"])
        z = np.polyfit(valid["wind_speed_mean"], valid[col], 1)
        x_fit = np.linspace(valid["wind_speed_mean"].min(),
                            valid["wind_speed_mean"].max(), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), "--", color=c, linewidth=2,
                alpha=0.7, label=f"{label}: r={corr:.2f}")

    ax.set_xlabel("Mean Wind Speed (m/s)")
    ax.set_ylabel("RMSE (C)")
    ax.set_title("Model Residuals vs Wind Speed: M2 (Derivative) vs M6 (Ground-Coupled)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rmse_vs_wind.png", bbox_inches="tight")
    print(f"Saved: rmse_vs_wind.png")
    plt.close()


def plot_cooling_gallery(df, results, n=9):
    """Fig: Grid of representative nights with M2 fits."""
    r = clean_results(results).sort_values("m6_tau")
    indices = np.linspace(0, len(r) - 1, n, dtype=int)
    selected = r.iloc[indices]

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(selected.iterrows()):
        ax = axes[i]
        night_date = row["night_date"]
        night_df = df[df["night_date"] == night_date]
        nd = prepare_night(night_df, eval_hours=9.0)

        if nd is None:
            ax.set_visible(False)
            continue

        t = nd["t"]
        valid = np.isfinite(nd["T_in"]) & np.isfinite(nd["T_out"])
        tv = t[valid]
        Ti = nd["T_in"][valid]
        To = nd["T_out"][valid]

        ax.plot(tv, Ti, "r-", linewidth=1.5, label="$T_{in}$")
        ax.plot(tv, To, "b-", linewidth=1.5, label="$T_{out}$")

        T_floor = nd["T_floor"]
        T_ground = nd.get("T_ground")
        T_ground_v = T_ground[valid] if T_ground is not None else None
        r6 = fit_model6(tv, Ti, To, T_ground=T_ground_v)
        if np.isfinite(r6.get("tau", np.nan)):
            ax.plot(tv, r6["pred"], "k--", linewidth=1.2, alpha=0.8,
                    label=f"M6: $\\tau$={r6['tau']:.1f}h, $\\alpha$={r6['alpha']:.2f}")
        else:
            r2 = fit_model2(tv, Ti, To)
            ax.plot(tv, r2["pred"], "k--", linewidth=1.2, alpha=0.8,
                    label=f"M2: $\\tau$={r2['tau']:.1f}h")

        ax.set_title(f"{night_date}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Hours", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r"Dome Cooling Curves: M2 Fits Across $\tau$ Range",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cooling_curves_gallery.png", bbox_inches="tight")
    print(f"Saved: cooling_curves_gallery.png")
    plt.close()


def plot_example_night(df, results, night_date=None):
    """Fig 6: Example night with all model fits."""
    if night_date is None:
        r = clean_results(results)
        median_tau = r["m6_tau"].median()
        idx = (r["m6_tau"] - median_tau).abs().idxmin()
        night_date = r.loc[idx, "night_date"]

    night_df = df[df["night_date"] == night_date]
    nd = prepare_night(night_df, eval_hours=9.0)
    if nd is None:
        print(f"Cannot prepare night {night_date}")
        return

    t = nd["t"]
    T_in = nd["T_in"]
    T_out = nd["T_out"]
    valid = np.isfinite(T_in) & np.isfinite(T_out)
    tv, Ti, To = t[valid], T_in[valid], T_out[valid]
    T_floor = nd["T_floor"]

    T_ground = nd.get("T_ground")
    T_ground_v = T_ground[valid] if T_ground is not None else None

    fits = {
        "M0: Exponential": (fit_model0(tv, Ti, To), "gray", 1.0),
        "M1: Basic ODE": (fit_model1(tv, Ti, To), "#8cb4d4", 1.5),
        "M2: Forced Ventilation": (fit_model2(tv, Ti, To), "#6baed6", 2.0),
        "M3: Stratified": (fit_model3(tv, Ti, To, T_ground=T_ground_v), "#9ecae1", 1.5),
        "M6: Stratified + Wind": (fit_model6(tv, Ti, To, T_ground=T_ground_v), "#2171b5", 2.5),
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={"hspace": 0.06})

    ax = axes[0]
    ax.plot(tv, nd["T_in_raw"][valid], color="lightcoral", alpha=0.3,
            linewidth=0.5, label="$T_{in}$ raw")
    ax.plot(tv, Ti, color="tab:red", linewidth=2.5, label="$T_{in}$ (smoothed)")
    ax.plot(tv, To, color="tab:blue", linewidth=2.5, label="$T_{out}$ (smoothed)")
    ax.axhline(T_floor, color="gray", linewidth=1, linestyle=":", alpha=0.5,
               label=f"$T_{{floor}}$ = {T_floor:.1f}C")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

    for name, (r, c, lw) in fits.items():
        parts = []
        if "tau" in r:
            parts.append(f"$\\tau$={r['tau']:.1f}h")
        if "C_buoy" in r:
            parts.append(f"C={r['C_buoy']:.1f}")
        if "alpha" in r:
            parts.append(f"$\\alpha$={r['alpha']:.2f}")
        if "k" in r:
            parts.append(f"k={r['k']:.2f}")
        label = f"{name} ({', '.join(parts)})"
        ax.plot(tv, r["pred"], "--", color=c, linewidth=lw, label=label)

    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Dome Thermalization: {night_date}", fontsize=14)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.05, "dome open", transform=ax.transAxes, fontsize=10,
            color="black", alpha=0.5)

    # Residuals
    ax = axes[1]
    for name, (r, c, lw) in fits.items():
        residuals = r["pred"] - Ti
        short = name.split(":")[0]
        ax.plot(tv, residuals, "-", color=c, linewidth=lw * 0.7, alpha=0.8,
                label=f"{short} RMSE={r['rmse']:.3f}")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
    ax.set_xlabel("Hours from dome open")
    ax.set_ylabel("Residual (C)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.savefig(FIGURES_DIR / "example_night_fit.png", bbox_inches="tight")
    print(f"Saved: example_night_fit.png (night={night_date})")
    plt.close()


def plot_delta_t_evolution(df):
    """Fig 7: Mean delta_T curve aligned at dome-open."""
    all_delta_t = []
    max_hours = 9

    for night in df["night_date"].unique():
        night_df = df[df["night_date"] == night]
        nd = prepare_night(night_df, eval_hours=max_hours)
        if nd is None:
            continue
        valid = np.isfinite(nd["T_in"]) & np.isfinite(nd["T_out"])
        t = nd["t"][valid]
        delta = nd["T_in"][valid] - nd["T_out"][valid]
        t_grid = np.arange(-1, max_hours, 1.0 / 60.0)
        delta_interp = np.interp(t_grid, t, delta, left=np.nan, right=np.nan)
        all_delta_t.append(delta_interp)

    stack = np.array(all_delta_t)
    t_hours = np.arange(-1, max_hours, 1.0 / 60.0)
    mean_dt = np.nanmean(stack, axis=0)
    p25 = np.nanpercentile(stack, 25, axis=0)
    p75 = np.nanpercentile(stack, 75, axis=0)
    p10 = np.nanpercentile(stack, 10, axis=0)
    p90 = np.nanpercentile(stack, 90, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_hours, mean_dt, "k-", linewidth=2.5, label="Mean", zorder=3)
    ax.fill_between(t_hours, p25, p75, color="steelblue", alpha=0.3,
                    label="25th--75th percentile", zorder=2)
    ax.fill_between(t_hours, p10, p90, color="steelblue", alpha=0.1,
                    label="10th--90th percentile", zorder=1)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
    ax.set_xlabel("Hours from Dome Open")
    ax.set_ylabel("$T_{in} - T_{out}$ (C)")
    ax.set_title(f"Dome $\\Delta T$ Evolution ({len(all_delta_t)} nights)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-1, max_hours)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delta_t_evolution.png", bbox_inches="tight")
    print(f"Saved: delta_t_evolution.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate dome thermal plots")
    parser.add_argument("--example-night", type=str, default=None)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_thermal_data()
    results = load_results()
    print(f"  {len(results)} nights in results")

    print("\nGenerating figures...")
    plot_tau_distribution(results)
    plot_tau_by_month(results)
    plot_model_comparison(results)
    plot_tau_vs_wind(results)
    plot_rmse_vs_wind(results)
    plot_cooling_gallery(df, results)
    plot_example_night(df, results, args.example_night)
    plot_delta_t_evolution(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
