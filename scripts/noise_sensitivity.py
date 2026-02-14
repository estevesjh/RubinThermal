#!/usr/bin/env python3
"""
Analyze how forecast noise affects thermal control performance.

Runs simulations across different temperature and rate noise levels
and generates comparison plots.
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
    NoisyForecastProvider,
)
from rubin_thermal.config import get_figures_path


def simulate_night(day, model, forecast_provider):
    """Simulate a single night with given forecast provider."""
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

        # Compute rate with forecast provider
        if forecast_provider is not None:
            rate = compute_rate(hours, temps, t, 1.0, forecast_provider, t)
            if hasattr(forecast_provider, 'get_rate_noise'):
                rate += forecast_provider.get_rate_noise(t, t, hours, temps)
        else:
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
    }


def run_simulation(test_days, model, temp_rmse_3h, rate_noise_std, seed=42):
    """Run simulation for all test days with given noise parameters."""
    # Convert 3h RMSE to per-hour rate
    rmse_per_hour = temp_rmse_3h / 3.0

    provider = NoisyForecastProvider(
        rmse_per_hour=rmse_per_hour,
        rate_noise_std=rate_noise_std,
        seed=seed,
    )

    all_overnight_errors = []
    sunset_errors = []

    for day in test_days:
        provider.clear_cache()  # Fresh noise for each night
        result = simulate_night(day, model, provider)
        sunset_errors.append(result["error_at_sunset"])
        all_overnight_errors.extend(result["overnight_errors"])

    overnight_errors = np.array(all_overnight_errors)
    sunset_errors = np.array(sunset_errors)

    return {
        "overnight_rms": np.sqrt(np.mean(overnight_errors ** 2)),
        "overnight_pct_good": 100 * np.mean(np.abs(overnight_errors) < 0.3),
        "sunset_rms": np.sqrt(np.mean(sunset_errors ** 2)),
        "sunset_pct_good": 100 * np.mean(np.abs(sunset_errors) < 0.3),
    }


def main():
    model = ThermalModel()
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Using {len(test_days)} test nights")

    # Noise levels to test
    temp_noise_levels = [0.0, 0.6, 0.8, 1.0]  # RMSE at 3h (°C)
    rate_noise_levels = [0.0, 0.1, 0.2, 0.3]  # Std dev (°C/hour)

    # Run simulations
    print("\nRunning noise sensitivity analysis...")
    results = {}

    for temp_noise in temp_noise_levels:
        for rate_noise in rate_noise_levels:
            key = (temp_noise, rate_noise)
            print(f"  Temp noise: {temp_noise}°C (3h), Rate noise: {rate_noise}°C/h...", end="", flush=True)
            results[key] = run_simulation(test_days, model, temp_noise, rate_noise)
            print(f" RMS={results[key]['overnight_rms']:.3f}°C")

    # Create results matrices
    rms_matrix = np.zeros((len(rate_noise_levels), len(temp_noise_levels)))
    pct_matrix = np.zeros((len(rate_noise_levels), len(temp_noise_levels)))

    for i, rate_noise in enumerate(rate_noise_levels):
        for j, temp_noise in enumerate(temp_noise_levels):
            rms_matrix[i, j] = results[(temp_noise, rate_noise)]["overnight_rms"]
            pct_matrix[i, j] = results[(temp_noise, rate_noise)]["overnight_pct_good"]

    # Figure 1: Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMS heatmap
    ax = axes[0]
    im = ax.imshow(rms_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(temp_noise_levels)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noise_levels])
    ax.set_yticks(range(len(rate_noise_levels)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_noise_levels])
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=11)
    ax.set_ylabel("Rate Noise Std Dev (°C/hour)", fontsize=11)
    ax.set_title("Overnight RMS Error (°C)", fontsize=12, fontweight="bold")

    # Add values to cells
    for i in range(len(rate_noise_levels)):
        for j in range(len(temp_noise_levels)):
            val = rms_matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="RMS (°C)")

    # Percent good heatmap
    ax = axes[1]
    im = ax.imshow(pct_matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(temp_noise_levels)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noise_levels])
    ax.set_yticks(range(len(rate_noise_levels)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_noise_levels])
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=11)
    ax.set_ylabel("Rate Noise Std Dev (°C/hour)", fontsize=11)
    ax.set_title("Overnight Within ±0.3°C (%)", fontsize=12, fontweight="bold")

    # Add values to cells
    for i in range(len(rate_noise_levels)):
        for j in range(len(temp_noise_levels)):
            val = pct_matrix[i, j]
            color = "white" if val < 45 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="Within ±0.3°C (%)")

    plt.suptitle("Forecast Noise Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "noise_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: noise_sensitivity_heatmap.png")
    plt.close()

    # Figure 2: Line plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMS vs temp noise (lines for each rate noise)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(rate_noise_levels)))
    for i, rate_noise in enumerate(rate_noise_levels):
        rms_values = [results[(t, rate_noise)]["overnight_rms"] for t in temp_noise_levels]
        ax.plot(temp_noise_levels, rms_values, "o-", color=colors[i],
                linewidth=2, markersize=8, label=f"Rate noise = {rate_noise:.2f}")

    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=11)
    ax.set_ylabel("Overnight RMS Error (°C)", fontsize=11)
    ax.set_title("RMS Error vs Temperature Noise", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Percent good vs temp noise
    ax = axes[1]
    for i, rate_noise in enumerate(rate_noise_levels):
        pct_values = [results[(t, rate_noise)]["overnight_pct_good"] for t in temp_noise_levels]
        ax.plot(temp_noise_levels, pct_values, "o-", color=colors[i],
                linewidth=2, markersize=8, label=f"Rate noise = {rate_noise:.2f}")

    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=11)
    ax.set_ylabel("Within ±0.3°C (%)", fontsize=11)
    ax.set_title("Performance vs Temperature Noise", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 70)

    plt.suptitle("Forecast Noise Impact on Control Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "noise_sensitivity_lines.png", dpi=150, bbox_inches="tight")
    print(f"Saved: noise_sensitivity_lines.png")
    plt.close()

    # Print summary table
    print("\n" + "=" * 70)
    print("NOISE SENSITIVITY RESULTS")
    print("=" * 70)
    print(f"{'Temp RMSE (3h)':<15} {'Rate Noise':<12} {'Overnight RMS':<15} {'Within ±0.3°C':<15}")
    print("-" * 70)
    for temp_noise in temp_noise_levels:
        for rate_noise in rate_noise_levels:
            r = results[(temp_noise, rate_noise)]
            print(f"{temp_noise:<15.1f} {rate_noise:<12.2f} {r['overnight_rms']:<15.3f} {r['overnight_pct_good']:<15.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
