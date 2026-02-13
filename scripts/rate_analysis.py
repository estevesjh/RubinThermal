#!/usr/bin/env python3
"""
Histogram of M1M3 temperature rate of change during nighttime.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main():
    model = ThermalModel()
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    # Load data
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Analyzing {len(test_days)} test nights")

    # Simulate and collect mirror temperature rates
    all_mirror_rates = []
    all_ambient_rates = []

    for day in test_days:
        hours = day["hours_from_sunset"]
        temps = day["temps"]
        T_sunset = day["T_sunset"]
        n = len(hours)

        T_setpoint = np.zeros(n)
        T_mirror = np.zeros(n)
        T_mirror[0] = T_sunset - model.cold_bias

        for i in range(n):
            t = hours[i]
            T_amb = temps[i]
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

        # Compute rates during overnight (t >= 0)
        overnight_mask = hours >= 0
        overnight_indices = np.where(overnight_mask)[0]

        for i in overnight_indices[1:]:  # Skip first point
            mirror_rate = (T_mirror[i] - T_mirror[i - 1]) / model.dt
            all_mirror_rates.append(mirror_rate)

            ambient_rate = (temps[i] - temps[i - 1]) / model.dt
            all_ambient_rates.append(ambient_rate)

    mirror_rates = np.array(all_mirror_rates)
    ambient_rates = np.array(all_ambient_rates)

    print(f"Total overnight rate samples: {len(mirror_rates)}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Mirror rate histogram
    ax = axes[0]
    bins = np.linspace(-2, 2, 81)
    ax.hist(mirror_rates, bins=bins, alpha=0.7, color="green", edgecolor="darkgreen", density=True)
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=np.mean(mirror_rates), color="red", linewidth=2, linestyle="--",
               label=f"Mean: {np.mean(mirror_rates):+.3f}C/hr")

    ax.set_xlabel("dT_mirror/dt (C/hour)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        "M1M3 Temperature Rate of Change (Overnight)\n"
        f"Std: {np.std(mirror_rates):.3f}C/hr",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)

    # Panel 2: Comparison with ambient rate
    ax = axes[1]
    ax.hist(ambient_rates, bins=bins, alpha=0.5, color="blue", edgecolor="navy",
            density=True, label=f"Ambient (std={np.std(ambient_rates):.3f}C/hr)")
    ax.hist(mirror_rates, bins=bins, alpha=0.5, color="green", edgecolor="darkgreen",
            density=True, label=f"Mirror (std={np.std(mirror_rates):.3f}C/hr)")
    ax.axvline(x=0, color="black", linewidth=2)

    ax.set_xlabel("dT/dt (C/hour)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Rate of Change: Mirror vs Ambient", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)

    plt.tight_layout()
    plt.savefig(figures_path / "mirror_rate_histogram.png", dpi=150, bbox_inches="tight")
    print("\nSaved: mirror_rate_histogram.png")
    plt.close()

    # Print statistics
    print("\n" + "=" * 60)
    print("M1M3 TEMPERATURE RATE OF CHANGE STATISTICS (overnight)")
    print("=" * 60)
    print(f"Mean:   {np.mean(mirror_rates):+.4f} C/hour")
    print(f"Std:    {np.std(mirror_rates):.4f} C/hour")
    print(f"Min:    {np.min(mirror_rates):+.4f} C/hour")
    print(f"Max:    {np.max(mirror_rates):+.4f} C/hour")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {np.percentile(mirror_rates, p):+.4f} C/hour")

    print(f"\n" + "=" * 60)
    print("FRACTION OF TIME AT VARIOUS RATES:")
    print("=" * 60)
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        pct_fast = 100 * np.mean(np.abs(mirror_rates) > thresh)
        print(f"  |rate| > {thresh:.1f}C/hr: {pct_fast:.2f}%")


if __name__ == "__main__":
    main()
