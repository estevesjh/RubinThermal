#!/usr/bin/env python3
"""
Analyze how forecast noise affects thermal control performance.

This script simulates Phase 2 (pre-sunset transition) where a forecast is
issued at T1 (-3h before sunset) and used throughout the night. The forecast
noise models realistic prediction uncertainty:

Forecast Usage:
- T_sunset_forecast: Predicted sunset temperature (3h lead time from T1)
- Rate: Computed from actual data, then noise added directly (rate_noise_std)

Noise Model:
- Temperature: noise_std = rmse_3h * (lead_time / 3h)
  - At 3h lead time: noise_std = rmse_3h
- Rate: noise_std = rate_noise_std (constant, independent of lead time)

Errors by Quantity:
- T_sunset (t=0): 3h lead time → noise_std = rmse_3h
- Rate: Direct noise with std = rate_noise_std (°C/hour)

Metric: Performance within ±0.5°C during first 3h after sunset (t=0 to t=3)
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

    # Get noisy forecast of sunset temperature (3h lead time from T1)
    if forecast_provider is not None:
        T_sunset_forecast = forecast_provider.get_temperature(0, T1, hours, temps)
    else:
        T_sunset_forecast = T_sunset

    t_sim = hours.copy()
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = np.zeros(n)

    T_mirror[0] = T_sunset_forecast - model.cold_bias

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb

        # Compute actual rate, then add noise (avoids overestimating error from differencing noisy temps)
        rate = compute_rate(hours, temps, t, 1.0)
        if forecast_provider is not None and hasattr(forecast_provider, 'get_rate_noise'):
            rate += forecast_provider.get_rate_noise(t, T1, hours, temps)

        if t < T1:
            T_setpoint[i] = T_sunset_forecast - model.cold_bias
        elif t < T2:
            alpha = (t - T1) / (T2 - T1)
            fixed = T_sunset_forecast - model.cold_bias
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

    # First 3 hours after sunset only
    first_3h_mask = (t_sim >= 0) & (t_sim <= 3)
    first_3h_errors = errors[first_3h_mask]

    return {
        "error_at_sunset": errors[sunset_idx],
        "overnight_errors": first_3h_errors,
    }


def run_simulation(test_days, model, temp_rmse_3h, rate_noise_std, seed=42):
    """Run simulation for all test days with given noise parameters."""
    provider = NoisyForecastProvider(
        rmse_per_hour=temp_rmse_3h,  # Now normalized at 3h in the provider
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
        "overnight_pct_good_05": 100 * np.mean(np.abs(overnight_errors) < 0.5),
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

    # Figure 2: Line plot - Within ±0.5°C vs temp noise, curves for rate noise
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(rate_noise_levels)))
    all_pct_values = []
    for i, rate_noise in enumerate(rate_noise_levels):
        pct_values = [results[(t, rate_noise)]["overnight_pct_good_05"] for t in temp_noise_levels]
        all_pct_values.extend(pct_values)
        ax.plot(temp_noise_levels, pct_values, "o-", color=colors[i],
                linewidth=2, markersize=8, label=f"Rate noise = {rate_noise:.2f} °C/h")

    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=11)
    ax.set_ylabel("Within ±0.5°C (%)", fontsize=11)
    ax.set_title("Control Performance vs Temp Noise (First 3h After Sunset)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(temp_noise_levels) + 0.05)
    ax.set_ylim(min(all_pct_values) - 3, max(all_pct_values) + 3)

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
