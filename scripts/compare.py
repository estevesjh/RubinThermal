#!/usr/bin/env python3
"""
Compare three-phase control algorithm to baseline strategies:
1. Match ambient (setpoint = ambient)
2. Ambient - 0.75C (fixed cold bias)
3. Three-phase control (our algorithm)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rubin_thermal import (
    CONFIG,
    load_temperature_data,
    build_day_database,
    train_test_split,
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    ThermalModel,
)
from rubin_thermal.config import get_figures_path


def simulate_control(day, strategy, model):
    """Simulate different control strategies."""
    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    t_sim = hours.copy()
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = temps.copy()

    # Initialize mirror
    if strategy == "three_phase":
        T_mirror[0] = T_sunset - model.cold_bias
    elif strategy == "match_ambient":
        T_mirror[0] = temps[0]
    else:  # ambient_minus_075
        T_mirror[0] = temps[0] - 0.75

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        rate = compute_rate(hours, temps, t, 1.0)

        if strategy == "match_ambient":
            T_setpoint[i] = T_amb

        elif strategy == "ambient_minus_075":
            T_setpoint[i] = T_amb - 0.75

        else:  # three_phase
            if t < T1:
                T_setpoint[i] = T_sunset - model.cold_bias
            elif t < T2:
                alpha = (t - T1) / (T2 - T1)
                fixed = T_sunset - model.cold_bias
                tracking = T_amb - model.cold_bias + 0.5 * model.tau * rate
                T_setpoint[i] = (1 - alpha) * fixed + alpha * tracking
            else:
                weights, times = make_linear_kernel()
                weighted_temp = sum(
                    w * interpolate_temp(hours, temps, t + dt)
                    for w, dt in zip(weights, times)
                )
                T_setpoint[i] = weighted_temp + 0.5 * model.tau * rate - model.cold_bias

        # Rate limiting
        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    # Compute errors relative to ambient
    errors_vs_ambient = T_mirror - T_ambient

    # Get overnight errors only (t >= 0)
    overnight_mask = t_sim >= 0
    overnight_errors = errors_vs_ambient[overnight_mask]

    return {
        "t": t_sim,
        "T_mirror": T_mirror,
        "T_ambient": T_ambient,
        "errors_vs_ambient": errors_vs_ambient,
        "overnight_errors": overnight_errors,
    }


def main():
    model = ThermalModel()
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_temperature_data()

    print("Building database...")
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Found {len(days)} days")
    print(f"Test set: {len(test_days)} nights")

    # Simulate all strategies
    print("\nSimulating all strategies...")

    strategies = {
        "Match Ambient": "match_ambient",
        "Ambient - 0.75C": "ambient_minus_075",
        "Three-Phase Control": "three_phase",
    }

    results = {name: [] for name in strategies.keys()}

    for day in test_days:
        for name, strategy in strategies.items():
            result = simulate_control(day, strategy, model)
            results[name].extend(result["overnight_errors"])

    # Convert to arrays
    for name in results:
        results[name] = np.array(results[name])
        print(f"{name}: {len(results[name])} samples")

    # Create comparison histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Match Ambient": "steelblue",
        "Ambient - 0.75C": "darkorange",
        "Three-Phase Control": "green",
    }

    # Panel 1: Overlaid histograms
    ax = axes[0, 0]
    bins = np.linspace(-1.5, 1.5, 61)

    for name, errs in results.items():
        ax.hist(errs, bins=bins, alpha=0.5, color=colors[name],
                label=f"{name}", density=True, edgecolor="none")

    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.1, color="green")

    ax.set_xlabel("T_mirror - T_ambient (C)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Overnight Error Distributions (Overlaid)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)

    # Panel 2: CDFs
    ax = axes[0, 1]

    for name, errs in results.items():
        sorted_err = np.sort(errs)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf * 100, linewidth=2.5, color=colors[name], label=name)

    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.1, color="green")
    ax.axhline(y=50, color="gray", linewidth=1, linestyle=":")

    ax.set_xlabel("T_mirror - T_ambient (C)", fontsize=12)
    ax.set_ylabel("Cumulative %", fontsize=12)
    ax.set_title("Cumulative Distribution Functions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 100)

    # Panel 3: Side-by-side histograms
    ax = axes[1, 0]

    width = 0.025
    bins_centers = np.arange(-1.2, 1.21, 0.075)
    bin_edges = np.arange(-1.2375, 1.25, 0.075)

    for i, (name, errs) in enumerate(results.items()):
        counts, _ = np.histogram(errs, bins=bin_edges, density=True)
        offset = (i - 1) * width
        ax.bar(bins_centers + offset, counts, width=width * 0.9, alpha=0.8,
               color=colors[name], label=name)

    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")

    ax.set_xlabel("T_mirror - T_ambient (C)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Side-by-Side Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-1.2, 1.2)

    # Panel 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    summary_lines = ["CONTROL STRATEGY COMPARISON", "=" * 50, ""]
    summary_lines.append(f"Test set: {len(test_days)} nights\n")

    for name, errs in results.items():
        mean = np.mean(errs)
        std = np.std(errs)
        rms = np.sqrt(np.mean(errs ** 2))
        pct_good = 100 * np.mean(np.abs(errs) < 0.3)
        pct_cold = 100 * np.mean(errs < 0)

        summary_lines.append(f"{name}:")
        summary_lines.append(f"  Mean:          {mean:+.3f}C")
        summary_lines.append(f"  Std dev:       {std:.3f}C")
        summary_lines.append(f"  RMS:           {rms:.3f}C")
        summary_lines.append(f"  Within +/-0.3C: {pct_good:.1f}%")
        summary_lines.append(f"  Mirror < Amb:  {pct_cold:.1f}%")
        summary_lines.append("")

    summary_lines.append("-" * 50)
    summary_lines.append("Target: Mirror should be BELOW ambient")
    summary_lines.append("        (negative error is good)")
    summary_lines.append("")
    summary_lines.append("Three-Phase aims for -0.3C bias")
    summary_lines.append("Ambient-0.75 aims for -0.75C bias")

    summary = "\n".join(summary_lines)

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Control Strategy Comparison: Overnight Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "control_comparison_histogram.png", dpi=150, bbox_inches="tight")
    print("\nSaved: control_comparison_histogram.png")
    plt.close()

    # Also create a cleaner single histogram comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(-1.5, 1.0, 51)

    for name, errs in results.items():
        ax.hist(errs, bins=bins, alpha=0.6, color=colors[name],
                label=f"{name} (RMS={np.sqrt(np.mean(errs**2)):.2f}C)",
                density=True, edgecolor="none")

    ax.axvline(x=0, color="black", linewidth=2.5, label="Ambient")
    ax.axvline(x=-0.3, color="darkgreen", linewidth=2, linestyle="--", label="Target (-0.3C)")
    ax.axvline(x=-0.75, color="darkorange", linewidth=2, linestyle=":", label="Target (-0.75C)")

    ax.set_xlabel("T_mirror - T_ambient (C)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.set_title(
        "Control Strategy Comparison: Mirror Temperature Relative to Ambient\n"
        f"{len(test_days)} nights, overnight only (sunset to sunrise)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.0)

    # Add annotation
    stats_text = []
    for name, errs in results.items():
        pct_good = 100 * np.mean(np.abs(errs) < 0.3)
        stats_text.append(f"{name}: {pct_good:.0f}% within +/-0.3C of target")

    ax.text(0.98, 0.95, "\n".join(stats_text), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(figures_path / "control_comparison_simple.png", dpi=150, bbox_inches="tight")
    print("Saved: control_comparison_simple.png")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
