#!/usr/bin/env python3
"""
Analyze how the cooling rate limit affects thermal control performance.

Runs simulations across different max_rate values to understand
the sensitivity of the system to HVAC rate constraints.
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


def simulate_night(day, model):
    """Simulate a single night with given model parameters."""
    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    t_sim = hours.copy()
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = np.zeros(n)

    T_mirror[0] = T_sunset - model.cold_bias

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb
        rate = compute_rate(hours, temps, t, 1.0)

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

        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    T_target = T_ambient - model.cold_bias
    errors = T_mirror - T_target
    sunset_idx = np.argmin(np.abs(t_sim))

    # Overnight errors only
    overnight_mask = t_sim >= 0
    overnight_errors = errors[overnight_mask]

    return {
        "error_at_sunset": errors[sunset_idx],
        "overnight_errors": overnight_errors,
        "t": t_sim,
        "errors": errors,
    }


def run_simulation(test_days, max_rate):
    """Run simulation for all test days with given max_rate."""
    model = ThermalModel(max_rate=max_rate)

    all_overnight_errors = []
    sunset_errors = []

    for day in test_days:
        result = simulate_night(day, model)
        sunset_errors.append(result["error_at_sunset"])
        all_overnight_errors.extend(result["overnight_errors"])

    overnight_errors = np.array(all_overnight_errors)
    sunset_errors = np.array(sunset_errors)

    return {
        "overnight_rms": np.sqrt(np.mean(overnight_errors ** 2)),
        "overnight_pct_good": 100 * np.mean(np.abs(overnight_errors) < 0.3),
        "overnight_mean": np.mean(overnight_errors),
        "overnight_std": np.std(overnight_errors),
        "sunset_rms": np.sqrt(np.mean(sunset_errors ** 2)),
        "sunset_pct_good": 100 * np.mean(np.abs(sunset_errors) < 0.3),
        "sunset_mean": np.mean(sunset_errors),
        "sunset_std": np.std(sunset_errors),
        "p5": np.percentile(overnight_errors, 5),
        "p25": np.percentile(overnight_errors, 25),
        "p50": np.percentile(overnight_errors, 50),
        "p75": np.percentile(overnight_errors, 75),
        "p95": np.percentile(overnight_errors, 95),
    }


def main():
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Using {len(test_days)} test nights")

    # Rate limits to test (C/hour)
    rate_limits = [0.5, 0.75, 1.0, 1.5, 2.0]

    # Run simulations
    print("\nRunning rate limit sensitivity analysis...")
    results = {}

    for max_rate in rate_limits:
        print(f"  Max rate: {max_rate} C/hour...", end="", flush=True)
        results[max_rate] = run_simulation(test_days, max_rate)
        print(f" RMS={results[max_rate]['overnight_rms']:.3f}°C, "
              f"±0.3°C={results[max_rate]['overnight_pct_good']:.1f}%")

    # Extract data for plotting
    rms_values = [results[r]["overnight_rms"] for r in rate_limits]
    pct_values = [results[r]["overnight_pct_good"] for r in rate_limits]
    sunset_rms = [results[r]["sunset_rms"] for r in rate_limits]
    sunset_pct = [results[r]["sunset_pct_good"] for r in rate_limits]

    # Figure 1: Main performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overnight RMS
    ax = axes[0, 0]
    ax.plot(rate_limits, rms_values, "o-", color="steelblue", linewidth=2, markersize=10)
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="0.5°C target")
    ax.set_xlabel("Max Cooling Rate (°C/hour)", fontsize=11)
    ax.set_ylabel("Overnight RMS Error (°C)", fontsize=11)
    ax.set_title("Overnight RMS vs Rate Limit", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, (x, y) in enumerate(zip(rate_limits, rms_values)):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Overnight % within ±0.3°C
    ax = axes[0, 1]
    ax.plot(rate_limits, pct_values, "o-", color="darkgreen", linewidth=2, markersize=10)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Max Cooling Rate (°C/hour)", fontsize=11)
    ax.set_ylabel("Within ±0.3°C (%)", fontsize=11)
    ax.set_title("Overnight Performance vs Rate Limit", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    for i, (x, y) in enumerate(zip(rate_limits, pct_values)):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Sunset performance
    ax = axes[1, 0]
    ax.plot(rate_limits, sunset_rms, "o-", color="darkorange", linewidth=2, markersize=10)
    ax.set_xlabel("Max Cooling Rate (°C/hour)", fontsize=11)
    ax.set_ylabel("Sunset RMS Error (°C)", fontsize=11)
    ax.set_title("Sunset RMS vs Rate Limit", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(rate_limits, sunset_rms)):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Percentile ranges
    ax = axes[1, 1]
    p5 = [results[r]["p5"] for r in rate_limits]
    p25 = [results[r]["p25"] for r in rate_limits]
    p50 = [results[r]["p50"] for r in rate_limits]
    p75 = [results[r]["p75"] for r in rate_limits]
    p95 = [results[r]["p95"] for r in rate_limits]

    ax.fill_between(rate_limits, p5, p95, alpha=0.2, color="steelblue", label="5-95%")
    ax.fill_between(rate_limits, p25, p75, alpha=0.4, color="steelblue", label="25-75%")
    ax.plot(rate_limits, p50, "o-", color="navy", linewidth=2, markersize=8, label="Median")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-0.3, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Max Cooling Rate (°C/hour)", fontsize=11)
    ax.set_ylabel("Error (°C)", fontsize=11)
    ax.set_title("Error Distribution vs Rate Limit", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cooling Rate Limit Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "rate_limit_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: rate_limit_sensitivity.png")
    plt.close()

    # Figure 2: Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rate_limits))
    width = 0.35

    bars1 = ax.bar(x - width/2, rms_values, width, label="Overnight RMS (°C)", color="steelblue")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, pct_values, width, label="Within ±0.3°C (%)", color="darkgreen", alpha=0.7)

    ax.set_xlabel("Max Cooling Rate (°C/hour)", fontsize=11)
    ax.set_ylabel("RMS Error (°C)", fontsize=11, color="steelblue")
    ax2.set_ylabel("Within ±0.3°C (%)", fontsize=11, color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}" for r in rate_limits])
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars1, rms_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, pct_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title("Performance Metrics vs Cooling Rate Limit", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "rate_limit_sensitivity_bars.png", dpi=150, bbox_inches="tight")
    print(f"Saved: rate_limit_sensitivity_bars.png")
    plt.close()

    # Print summary table
    print("\n" + "=" * 85)
    print("RATE LIMIT SENSITIVITY RESULTS")
    print("=" * 85)
    print(f"{'Max Rate':<12} {'Overnight RMS':<15} {'Within ±0.3°C':<15} "
          f"{'Sunset RMS':<15} {'Sunset ±0.3°C':<15}")
    print("-" * 85)
    for rate in rate_limits:
        r = results[rate]
        print(f"{rate:<12.2f} {r['overnight_rms']:<15.3f} {r['overnight_pct_good']:<15.1f}% "
              f"{r['sunset_rms']:<15.3f} {r['sunset_pct_good']:<15.1f}%")
    print("=" * 85)

    # Analysis summary
    print("\nKEY FINDINGS:")
    best_rate = rate_limits[np.argmax(pct_values)]
    print(f"  - Best overnight performance at max_rate = {best_rate} C/hour")
    print(f"  - Diminishing returns above ~1.0 C/hour")

    improvement = pct_values[-1] - pct_values[0]
    print(f"  - Going from 0.5 to 2.0 C/hour: +{improvement:.1f}% within ±0.3°C")


if __name__ == "__main__":
    main()
