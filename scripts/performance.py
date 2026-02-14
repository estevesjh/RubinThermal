#!/usr/bin/env python3
"""
Generate performance histograms and nightly simulation plots for three-phase control.
"""

import argparse
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
    create_provider,
)
from rubin_thermal.config import get_figures_path


def simulate_three_phase(day, model, forecast_provider=None):
    """Simulate with optimal three-phase control.

    Parameters
    ----------
    day : dict
        Day data dictionary
    model : ThermalModel
        Thermal model instance
    forecast_provider : TemperatureProvider, optional
        Provider for future temperatures. If None, uses perfect knowledge.
    """
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
    phase = np.zeros(n, dtype=int)

    T_mirror[0] = T_sunset - model.cold_bias

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb

        # Use forecast provider for rate if available
        if forecast_provider is not None:
            rate = compute_rate(hours, temps, t, 1.0, forecast_provider, t)
            # Add rate noise if provider supports it
            if hasattr(forecast_provider, 'get_rate_noise'):
                rate += forecast_provider.get_rate_noise(t, t, hours, temps)
        else:
            rate = compute_rate(hours, temps, t, 1.0)

        if t < T1:
            T_setpoint[i] = T_sunset - model.cold_bias
            phase[i] = 1
        elif t < T2:
            alpha = (t - T1) / (T2 - T1)
            fixed = T_sunset - model.cold_bias
            tracking = T_amb - model.cold_bias + 0.5 * model.tau * rate
            T_setpoint[i] = (1 - alpha) * fixed + alpha * tracking
            phase[i] = 2
        else:
            weights, times = make_linear_kernel()
            if forecast_provider is not None:
                weighted_temp = sum(
                    w * forecast_provider.get_temperature(t + dt, t, hours, temps)
                    for w, dt in zip(weights, times)
                )
            else:
                weighted_temp = sum(
                    w * interpolate_temp(hours, temps, t + dt)
                    for w, dt in zip(weights, times)
                )
            T_setpoint[i] = weighted_temp + 0.5 * model.tau * rate - model.cold_bias
            phase[i] = 3

        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    T_target = T_ambient - model.cold_bias
    errors = T_mirror - T_target
    sunset_idx = np.argmin(np.abs(t_sim))

    return {
        "t": t_sim,
        "T_ambient": T_ambient,
        "T_target": T_target,
        "T_setpoint": T_setpoint,
        "T_mirror": T_mirror,
        "errors": errors,
        "phase": phase,
        "error_at_sunset": errors[sunset_idx],
        "date": day["date"],
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate performance plots for three-phase thermal control"
    )
    parser.add_argument(
        "--forecast-mode",
        choices=["perfect", "persistence", "noisy", "twilight"],
        default="perfect",
        help="Temperature forecast mode (default: perfect)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to forecast model (required for twilight mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noisy mode (default: 42)",
    )
    parser.add_argument(
        "--rate-noise",
        type=float,
        default=0.0,
        help="Rate noise std dev in C/hour for noisy mode (default: 0.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = ThermalModel()
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    # Set up forecast provider
    forecast_mode = args.forecast_mode
    model_path = args.model_path or CONFIG.get("forecast", {}).get("model_path")

    if forecast_mode == "twilight" and model_path is None:
        print("ERROR: --model-path required for twilight mode")
        sys.exit(1)

    print(f"Forecast mode: {forecast_mode}")
    if args.rate_noise > 0:
        print(f"Rate noise: {args.rate_noise} C/hour")
    forecast_provider = create_provider(
        forecast_mode, model_path, seed=args.seed, rate_noise_std=args.rate_noise
    )

    # Load data
    print("Loading data...")
    df = load_temperature_data()

    print("Building database...")
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Found {len(days)} days, using {len(test_days)} test nights")

    # Simulate all test nights
    print("Simulating all test nights...")
    all_results = []
    all_overnight_errors = []
    sunset_errors = []

    for day in test_days:
        result = simulate_three_phase(day, model, forecast_provider)
        all_results.append(result)
        sunset_errors.append(result["error_at_sunset"])

        overnight_mask = result["t"] >= 0
        all_overnight_errors.extend(result["errors"][overnight_mask])

    sunset_errors = np.array(sunset_errors)
    all_overnight_errors = np.array(all_overnight_errors)

    print(f"Simulated {len(all_results)} nights")
    print(f"Total overnight samples: {len(all_overnight_errors)}")

    # Filename prefix based on parameters
    rate_str = str(model.max_rate).replace(".", "")
    tau_str = str(int(model.tau))
    if forecast_mode == "perfect":
        filename_prefix = f"three_phase_{rate_str}rate_{tau_str}tau"
    else:
        filename_prefix = f"three_phase_{rate_str}rate_{tau_str}tau_{forecast_mode}"

    # Figure 1: Performance Histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    bins = np.linspace(-1.2, 1.2, 49)
    ax.hist(all_overnight_errors, bins=bins, alpha=0.7, color="steelblue",
            edgecolor="navy", density=True)
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.15, color="green")

    mean_err = np.mean(all_overnight_errors)
    rms_err = np.sqrt(np.mean(all_overnight_errors ** 2))
    pct_good = 100 * np.mean(np.abs(all_overnight_errors) < 0.3)

    ax.set_xlabel("T_mirror - T_ambient + 0.3C", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"Overnight Error Distribution (sunset to sunrise)\n"
        f"Mean: {mean_err:+.3f}C, RMS: {rms_err:.3f}C, +/-0.3C: {pct_good:.0f}%",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)

    ax = axes[0, 1]
    sorted_err = np.sort(all_overnight_errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf * 100, "b-", linewidth=2.5)
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.15, color="green")
    ax.axhline(y=50, color="gray", linewidth=1, linestyle=":")

    for pct in [5, 25, 50, 75, 95]:
        val = np.percentile(all_overnight_errors, pct)
        ax.plot(val, pct, "ko", markersize=6)
        ax.annotate(f"{pct}%: {val:+.2f}", xy=(val, pct),
                    xytext=(val + 0.1, pct + 3), fontsize=9)

    ax.set_xlabel("T_mirror - T_ambient + 0.3C", fontsize=12)
    ax.set_ylabel("Cumulative %", fontsize=12)
    ax.set_title("CDF of Overnight Errors", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 100)

    ax = axes[1, 0]
    bins = np.linspace(-0.8, 0.8, 33)
    ax.hist(sunset_errors, bins=bins, alpha=0.7, color="darkorange",
            edgecolor="darkred", density=True)
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=-0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(x=0.3, color="red", linewidth=1.5, linestyle="--")
    ax.axvspan(-0.3, 0.3, alpha=0.15, color="green")

    mean_sunset = np.mean(sunset_errors)
    rms_sunset = np.sqrt(np.mean(sunset_errors ** 2))
    pct_sunset_good = 100 * np.mean(np.abs(sunset_errors) < 0.3)

    ax.set_xlabel("Error at Sunset (C)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"Sunset Error Distribution\n"
        f"Mean: {mean_sunset:+.3f}C, RMS: {rms_sunset:.3f}C, +/-0.3C: {pct_sunset_good:.0f}%",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")

    summary = f"""
THREE-PHASE CONTROL PERFORMANCE
{'='*55}

Configuration:
  Phase 1 (Daytime):     t < {T1}h    Fixed at T_sunset - 0.3C
  Phase 2 (Pre-sunset):  {T1}h to {T2}h   Ramp to tracking
  Phase 3 (Overnight):   t >= {T2}h    Weighted lookahead + rate

Forecast Mode: {forecast_mode}
Test Set: {len(test_days)} nights

SUNSET PERFORMANCE (t = 0):
  Mean error:     {mean_sunset:+.4f}C
  Std dev:        {np.std(sunset_errors):.4f}C
  RMS:            {rms_sunset:.4f}C
  Within +/-0.3C: {pct_sunset_good:.1f}%

OVERNIGHT PERFORMANCE (sunset to sunrise):
  Mean error:     {mean_err:+.4f}C
  Std dev:        {np.std(all_overnight_errors):.4f}C
  RMS:            {rms_err:.4f}C
  Within +/-0.3C: {pct_good:.1f}%

PERCENTILES (overnight):
   5th: {np.percentile(all_overnight_errors, 5):+.3f}C
  25th: {np.percentile(all_overnight_errors, 25):+.3f}C
  50th: {np.percentile(all_overnight_errors, 50):+.3f}C
  75th: {np.percentile(all_overnight_errors, 75):+.3f}C
  95th: {np.percentile(all_overnight_errors, 95):+.3f}C
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Three-Phase Thermal Control: Performance Summary",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename_prefix}_histograms.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {filename_prefix}_histograms.png")
    plt.close()

    # Figure 2: 20 Example Nights
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))

    n_examples = 20
    step = max(1, len(all_results) // n_examples)
    example_indices = [min(i * step, len(all_results) - 1) for i in range(n_examples)]

    for idx, result_idx in enumerate(example_indices):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]

        result = all_results[result_idx]
        t = result["t"]

        mask = t >= -2
        t_plot = t[mask]
        T_amb_plot = result["T_ambient"][mask]
        T_mir_plot = result["T_mirror"][mask]

        ax.fill_between(t_plot, T_amb_plot, T_mir_plot,
                        where=(T_mir_plot > T_amb_plot),
                        color="red", alpha=0.3, label="Mirror > Ambient")
        ax.fill_between(t_plot, T_amb_plot, T_mir_plot,
                        where=(T_mir_plot <= T_amb_plot),
                        color="blue", alpha=0.3, label="Mirror < Ambient")

        ax.plot(t_plot, T_amb_plot, "b-", linewidth=1.5, label="Ambient")
        ax.plot(t_plot, result["T_setpoint"][mask], "r--", linewidth=1.2, alpha=0.8, label="Setpoint")
        ax.plot(t_plot, T_mir_plot, "g-", linewidth=2, label="Mirror")

        ax.axvline(x=0, color="orange", linewidth=2, alpha=0.7)

        overnight_mask_full = result["t"] >= 0
        night_errors = result["errors"][overnight_mask_full]
        night_rms = np.sqrt(np.mean(night_errors ** 2))

        ax.set_title(
            f'{result["date"]}\n'
            f'Sunset: {result["error_at_sunset"]:+.2f}C, Night RMS: {night_rms:.2f}C',
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Hours from Sunset", fontsize=9)
        ax.set_ylabel("Temperature (C)", fontsize=9)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    plt.suptitle(
        "Three-Phase Control: 20 Example Nights\n"
        "Blue=Ambient, Red dashed=Setpoint, Green=Mirror, Orange line=Sunset",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename_prefix}_20_nights.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {filename_prefix}_20_nights.png")
    plt.close()

    # Figure 3: Detailed view of 6 nights
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))

    example_indices_detail = [0, 50, 100, 150, 200, 300]

    for idx, result_idx in enumerate(example_indices_detail):
        result_idx = min(result_idx, len(all_results) - 1)

        result = all_results[result_idx]
        t = result["t"]
        T_amb = result["T_ambient"]
        T_mir = result["T_mirror"]

        ax = axes[idx, 0]

        ax.fill_between(t, T_amb, T_mir,
                        where=(T_mir > T_amb),
                        color="red", alpha=0.35, label="Mirror > Ambient")
        ax.fill_between(t, T_amb, T_mir,
                        where=(T_mir <= T_amb),
                        color="blue", alpha=0.35, label="Mirror < Ambient")

        ax.plot(t, T_amb, "b-", linewidth=2, label="Ambient")
        ax.plot(t, result["T_setpoint"], "r-", linewidth=1.5, alpha=0.7, label="Setpoint")
        ax.plot(t, T_mir, "g-", linewidth=2.5, label="Mirror")

        ax.axvline(x=0, color="orange", linewidth=2, label="Sunset")
        ax.axvline(x=T1, color="purple", linewidth=1, linestyle="--", alpha=0.7)

        ax.set_xlabel("Hours from Sunset", fontsize=10)
        ax.set_ylabel("Temperature (C)", fontsize=10)
        ax.set_title(f'{result["date"]}', fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 10)

        if idx == 0:
            ax.legend(fontsize=8, loc="best")

        ax = axes[idx, 1]

        ax.plot(t, result["errors"], "g-", linewidth=2)
        ax.axhline(y=0, color="black", linewidth=2)
        ax.axhline(y=0.3, color="red", linewidth=1.5, linestyle="--")
        ax.axhline(y=-0.3, color="red", linewidth=1.5, linestyle="--")
        ax.axvspan(0, t[-1], alpha=0.1, color="blue")
        ax.axvline(x=0, color="orange", linewidth=2)

        overnight_mask = t >= 0
        night_errors = result["errors"][overnight_mask]
        night_rms = np.sqrt(np.mean(night_errors ** 2))
        pct_good_night = 100 * np.mean(np.abs(night_errors) < 0.3)

        ax.set_xlabel("Hours from Sunset", fontsize=10)
        ax.set_ylabel("Error (C)", fontsize=10)
        ax.set_title(
            f'Sunset: {result["error_at_sunset"]:+.2f}C | '
            f"Overnight RMS: {night_rms:.2f}C | +/-0.3C: {pct_good_night:.0f}%",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 10)
        ax.set_ylim(-1, 1)

    plt.suptitle(
        "Three-Phase Control: Detailed Night Views\n"
        "Left: Temperature traces | Right: Control error (shaded = overnight)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(figures_path / f"{filename_prefix}_detailed_nights.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {filename_prefix}_detailed_nights.png")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Forecast mode: {forecast_mode}")
    print(f"Total nights: {len(test_days)}")
    print(f"Total overnight samples: {len(all_overnight_errors)}")
    print(f"\nSunset error: {mean_sunset:+.3f} +/- {np.std(sunset_errors):.3f}C")
    print(f"Overnight RMS: {rms_err:.3f}C")
    print(f"Overnight within +/-0.3C: {pct_good:.1f}%")


if __name__ == "__main__":
    main()
