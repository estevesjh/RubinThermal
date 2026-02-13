#!/usr/bin/env python3
"""
Three-Phase Control Optimization

Grid search over phase2 algorithms and transition times (T1, T2).
Train/test split: even dates train, odd dates test.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
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
    ThermalModel,
    phase1_fixed,
    phase3_lookahead,
)
from rubin_thermal.control import PHASE2_OPTIONS
from rubin_thermal.config import get_figures_path


def simulate_three_phase(day, T1, T2, phase1_func, phase2_func, phase3_func, model):
    """Simulate three-phase control for one day."""
    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    t_sim = hours.copy()
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = np.zeros(n)
    phase = np.zeros(n, dtype=int)

    # Initialize mirror at daytime setpoint
    T_mirror[0] = phase1_func(
        t_sim[0], temps[0], T_sunset, 0, hours, temps, model.cold_bias
    )

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb
        rate = compute_rate(hours, temps, t, 1.0)

        if t < T1:
            T_setpoint[i] = phase1_func(
                t, T_amb, T_sunset, rate, hours, temps, model.cold_bias
            )
            phase[i] = 1
        elif t < T2:
            T_setpoint[i] = phase2_func(
                t, T_amb, T_sunset, rate, hours, temps, T1, T2, model.tau, model.cold_bias
            )
            phase[i] = 2
        else:
            T_setpoint[i] = phase3_func(
                t, T_amb, T_sunset, rate, hours, temps, model.tau, model.cold_bias
            )
            phase[i] = 3

        # Rate limiting
        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])

        # Thermal dynamics
        if i > 0:
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    T_target = T_ambient - model.cold_bias
    errors = T_mirror - T_target

    sunset_idx = np.argmin(np.abs(t_sim))
    error_at_sunset = errors[sunset_idx]

    overnight_mask = t_sim >= 0
    rms_overnight = (
        np.sqrt(np.mean(errors[overnight_mask] ** 2))
        if np.any(overnight_mask)
        else np.nan
    )

    return {
        "t": t_sim,
        "T_ambient": T_ambient,
        "T_target": T_target,
        "T_setpoint": T_setpoint,
        "T_mirror": T_mirror,
        "errors": errors,
        "phase": phase,
        "error_at_sunset": error_at_sunset,
        "rms_overnight": rms_overnight,
    }


def main():
    # Load configuration
    model = ThermalModel()
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_temperature_data()

    # Build database
    print("Building day+night database...")
    days = build_day_database(df)
    train_days, test_days = train_test_split(days)
    print(f"Found {len(days)} days")
    print(f"Train: {len(train_days)}, Test: {len(test_days)}")

    # Grid search parameters
    T1_options = [-6, -5, -4, -3, -2]
    T2_options = [-2, -1, 0]

    print("\n" + "=" * 70)
    print("THREE-PHASE OPTIMIZATION")
    print("=" * 70)
    print(
        f"\nTesting {len(PHASE2_OPTIONS)} phase2 algorithms x "
        f"{len(T1_options)} T1 values x {len(T2_options)} T2 values"
    )

    results = []

    for phase2_name, phase2_func in PHASE2_OPTIONS.items():
        for T1 in T1_options:
            for T2 in T2_options:
                if T2 <= T1:
                    continue

                sunset_errors = []
                overnight_rms = []

                for day in train_days:
                    result = simulate_three_phase(
                        day, T1, T2, phase1_fixed, phase2_func, phase3_lookahead, model
                    )
                    sunset_errors.append(result["error_at_sunset"])
                    if not np.isnan(result["rms_overnight"]):
                        overnight_rms.append(result["rms_overnight"])

                sunset_errors = np.array(sunset_errors)
                overnight_rms = np.array(overnight_rms)

                results.append({
                    "phase2": phase2_name,
                    "T1": T1,
                    "T2": T2,
                    "phase2_func": phase2_func,
                    "sunset_mean": np.mean(sunset_errors),
                    "sunset_rms": np.sqrt(np.mean(sunset_errors ** 2)),
                    "overnight_rms": np.mean(overnight_rms),
                    "pct_sunset_good": 100 * np.mean(np.abs(sunset_errors) < 0.3),
                })

    results.sort(key=lambda x: x["sunset_rms"] + x["overnight_rms"])

    print("\nTop 15 configurations (training set):")
    print("-" * 90)
    print(
        f"{'Rank':<5} {'Phase2 Algorithm':<20} {'T1':>4} {'T2':>4} "
        f"{'Sunset RMS':>11} {'Overnight':>10} {'Total':>8}"
    )
    print("-" * 90)
    for i, r in enumerate(results[:15]):
        total = r["sunset_rms"] + r["overnight_rms"]
        print(
            f"{i+1:<5} {r['phase2']:<20} {r['T1']:>4}h {r['T2']:>4}h "
            f"{r['sunset_rms']:>10.3f}C {r['overnight_rms']:>9.3f}C {total:>7.3f}"
        )

    # Test set evaluation
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (Top 5)")
    print("=" * 70)

    test_results = []
    for r in results[:5]:
        sunset_errors = []
        overnight_rms = []
        all_errors = []

        for day in test_days:
            result = simulate_three_phase(
                day, r["T1"], r["T2"], phase1_fixed, r["phase2_func"], phase3_lookahead, model
            )
            sunset_errors.append(result["error_at_sunset"])
            if not np.isnan(result["rms_overnight"]):
                overnight_rms.append(result["rms_overnight"])
            overnight_mask = result["t"] >= 0
            all_errors.extend(result["errors"][overnight_mask])

        sunset_errors = np.array(sunset_errors)
        all_errors = np.array(all_errors)

        test_results.append({
            **r,
            "test_sunset_mean": np.mean(sunset_errors),
            "test_sunset_rms": np.sqrt(np.mean(sunset_errors ** 2)),
            "test_overnight_rms": np.mean(overnight_rms),
            "test_pct_sunset_good": 100 * np.mean(np.abs(sunset_errors) < 0.3),
            "test_all_errors": all_errors,
            "test_sunset_errors": sunset_errors,
        })

        print(f"{r['phase2']:<20} T1={r['T1']}h, T2={r['T2']}h")
        print(
            f"  Sunset: {np.mean(sunset_errors):+.3f}+/-{np.std(sunset_errors):.3f}C, "
            f"within +/-0.3C: {100*np.mean(np.abs(sunset_errors)<0.3):.0f}%"
        )
        print(f"  Overnight RMS: {np.mean(overnight_rms):.3f}C")
        print()

    best = test_results[0]

    # Create visualization
    rate_str = str(model.max_rate).replace(".", "")
    tau_str = str(int(model.tau))
    filename_prefix = f"three_phase_control_{rate_str}rate_{tau_str}tau"

    # Figure 1: Example day
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    example_day = test_days[min(25, len(test_days) - 1)]
    result = simulate_three_phase(
        example_day, best["T1"], best["T2"], phase1_fixed, best["phase2_func"], phase3_lookahead, model
    )

    ax = axes[0, 0]
    ax.plot(result["t"], result["T_ambient"], "b-", linewidth=2, label="Ambient")
    ax.plot(result["t"], result["T_target"], "k--", linewidth=1.5, label="Target (Amb-0.3)")
    ax.plot(result["t"], result["T_setpoint"], "r-", linewidth=1.5, alpha=0.7, label="Setpoint")
    ax.plot(result["t"], result["T_mirror"], "g-", linewidth=2.5, label="Mirror")

    ax.axvline(x=0, color="orange", linewidth=2, linestyle="-", label="Sunset")
    ax.axvline(x=best["T1"], color="purple", linewidth=1.5, linestyle="--", label=f'T1={best["T1"]}h')
    ax.axvline(x=best["T2"], color="cyan", linewidth=1.5, linestyle="--", label=f'T2={best["T2"]}h')

    colors = ["yellow", "orange", "lightblue"]
    t = result["t"]
    for p in [1, 2, 3]:
        mask = result["phase"] == p
        if np.any(mask):
            ax.axvspan(t[mask][0], t[mask][-1], alpha=0.15, color=colors[p - 1])

    ax.set_xlabel("Hours from Sunset", fontsize=11)
    ax.set_ylabel("Temperature (C)", fontsize=11)
    ax.set_title(
        f'Three-Phase Control: {example_day["date"]}\n'
        f'Sunset error: {result["error_at_sunset"]:+.2f}C',
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 8)

    ax = axes[0, 1]
    ax.plot(result["t"], result["errors"], "g-", linewidth=2)
    ax.axhline(y=0, color="black", linewidth=2)
    ax.axhline(y=0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axhline(y=-0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.1, color="green")
    ax.axvline(x=0, color="orange", linewidth=2)
    ax.axvline(x=best["T1"], color="purple", linewidth=1.5, linestyle="--")
    ax.axvline(x=best["T2"], color="cyan", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Hours from Sunset", fontsize=11)
    ax.set_ylabel("Error: T_mirror - T_ambient + 0.3 (C)", fontsize=11)
    ax.set_title("Control Error", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 8)
    ax.set_ylim(-1, 1)

    ax = axes[1, 0]
    ax.hist(
        best["test_sunset_errors"],
        bins=30,
        alpha=0.7,
        color="steelblue",
        edgecolor="navy",
        density=True,
    )
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axvline(x=-0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.15, color="green")

    ax.set_xlabel("Error at Sunset (C)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f'Distribution of Sunset Errors\n'
        f'Mean: {np.mean(best["test_sunset_errors"]):+.3f}C, '
        f'+/-0.3C: {best["test_pct_sunset_good"]:.0f}%',
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(
        best["test_all_errors"],
        bins=50,
        alpha=0.7,
        color="green",
        edgecolor="darkgreen",
        density=True,
    )
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axvline(x=-0.3, color="gray", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.15, color="green")

    pct_good = 100 * np.mean(np.abs(best["test_all_errors"]) < 0.3)
    ax.set_xlabel("Overnight Error (C)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f'Distribution of Overnight Errors\n'
        f'RMS: {best["test_overnight_rms"]:.3f}C, +/-0.3C: {pct_good:.0f}%',
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)

    plt.suptitle(
        f'Best Three-Phase Control: {best["phase2"]}, T1={best["T1"]}h, T2={best["T2"]}h',
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename_prefix}_best.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {filename_prefix}_best.png")
    plt.close()

    # Figure 2: Multiple example nights
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for idx in range(9):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        day_idx = min(idx * 40, len(test_days) - 1)
        day = test_days[day_idx]
        result = simulate_three_phase(
            day, best["T1"], best["T2"], phase1_fixed, best["phase2_func"], phase3_lookahead, model
        )

        ax.plot(result["t"], result["T_ambient"], "b-", linewidth=1.5, label="Ambient")
        ax.plot(result["t"], result["T_target"], "k--", linewidth=1, label="Target")
        ax.plot(result["t"], result["T_mirror"], "g-", linewidth=2, label="Mirror")

        ax.axvline(x=0, color="orange", linewidth=2)
        ax.axvline(x=best["T1"], color="purple", linewidth=1, linestyle="--")
        ax.axvline(x=best["T2"], color="cyan", linewidth=1, linestyle="--")

        for p, color in [(1, "yellow"), (2, "orange"), (3, "lightblue")]:
            mask = result["phase"] == p
            if np.any(mask):
                ax.axvspan(result["t"][mask][0], result["t"][mask][-1], alpha=0.1, color=color)

        ax.set_title(
            f'{day["date"]}\nSunset err: {result["error_at_sunset"]:+.2f}C, '
            f'Night RMS: {result["rms_overnight"]:.2f}C',
            fontsize=10,
        )
        ax.set_xlabel("Hours from Sunset", fontsize=9)
        ax.set_ylabel("Temp (C)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 8)

        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle(
        f'Three-Phase Control: Multiple Nights\n'
        f'{best["phase2"]}, T1={best["T1"]}h, T2={best["T2"]}h',
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename_prefix}_examples.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {filename_prefix}_examples.png")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMAL THREE-PHASE CONTROL SYSTEM")
    print("=" * 70)
    print(
        f"""
PHASE 1: DAYTIME (t < {best['T1']}h before sunset)
  Algorithm: Fixed setpoint at T_predicted_sunset - 0.3C

PHASE 2: PRE-SUNSET TRANSITION ({best['T1']}h <= t < {best['T2']}h)
  Algorithm: {best['phase2']}
  Duration: {abs(best['T2'] - best['T1'])} hours

PHASE 3: OVERNIGHT (t >= {best['T2']}h)
  Algorithm: Linear weighted lookahead (0-3h) + 0.5*tau*rate - 0.3C

PERFORMANCE (test set, {len(test_days)} nights):
  Sunset error: {np.mean(best['test_sunset_errors']):+.3f} +/- {np.std(best['test_sunset_errors']):.3f}C
  Sunset +/-0.3C: {best['test_pct_sunset_good']:.0f}%
  Overnight RMS: {best['test_overnight_rms']:.3f}C
  Overnight +/-0.3C: {100*np.mean(np.abs(best['test_all_errors']) < 0.3):.0f}%
"""
    )


if __name__ == "__main__":
    main()
