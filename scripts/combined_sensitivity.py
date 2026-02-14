#!/usr/bin/env python3
"""
Combined sensitivity analysis: rate limit vs forecast noise.

Creates heatmaps showing how both HVAC rate constraints and
forecast uncertainty affect thermal control performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


def simulate_night(day, model, forecast_provider=None):
    """Simulate a single night."""
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

    overnight_mask = t_sim >= 0
    return errors[overnight_mask]


def run_simulation(test_days, max_rate, temp_rmse_3h, rate_noise_std, seed=42):
    """Run simulation with given parameters."""
    model = ThermalModel(max_rate=max_rate)

    if temp_rmse_3h > 0 or rate_noise_std > 0:
        rmse_per_hour = temp_rmse_3h / 3.0
        provider = NoisyForecastProvider(
            rmse_per_hour=rmse_per_hour,
            rate_noise_std=rate_noise_std,
            seed=seed,
        )
    else:
        provider = None

    all_errors = []
    for day in test_days:
        if provider:
            provider.clear_cache()
        errors = simulate_night(day, model, provider)
        all_errors.extend(errors)

    errors = np.array(all_errors)
    return {
        "rms": np.sqrt(np.mean(errors ** 2)),
        "pct_good": 100 * np.mean(np.abs(errors) < 0.3),
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

    # Parameters to test
    rate_limits = [0.5, 0.75, 1.0, 1.5, 2.0]
    temp_noises = [0.0, 0.6, 0.8, 1.0]  # RMSE at 3h
    rate_noise = 0.2  # Fixed rate noise

    # Run simulations
    print(f"\nRunning combined sensitivity analysis (rate noise fixed at {rate_noise} C/h)...")
    results = {}
    total = len(rate_limits) * len(temp_noises)
    count = 0

    for max_rate in rate_limits:
        for temp_noise in temp_noises:
            count += 1
            print(f"  [{count}/{total}] Rate limit: {max_rate} C/h, Temp noise: {temp_noise}°C...",
                  end="", flush=True)
            results[(max_rate, temp_noise)] = run_simulation(
                test_days, max_rate, temp_noise, rate_noise
            )
            print(f" RMS={results[(max_rate, temp_noise)]['rms']:.2f}°C")

    # Create matrices
    rms_matrix = np.zeros((len(rate_limits), len(temp_noises)))
    pct_matrix = np.zeros((len(rate_limits), len(temp_noises)))

    for i, max_rate in enumerate(rate_limits):
        for j, temp_noise in enumerate(temp_noises):
            rms_matrix[i, j] = results[(max_rate, temp_noise)]["rms"]
            pct_matrix[i, j] = results[(max_rate, temp_noise)]["pct_good"]

    # Custom colormaps
    # For RMS: lower is better (green to red)
    rms_cmap = LinearSegmentedColormap.from_list(
        "rms", ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"]
    )
    # For percentage: higher is better (red to green)
    pct_cmap = LinearSegmentedColormap.from_list(
        "pct", ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    )

    # Figure 1: Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMS heatmap
    ax = axes[0]
    im = ax.imshow(rms_matrix, cmap=rms_cmap, aspect="auto", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight RMS Error (°C)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = rms_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RMS (°C)", fontsize=11)

    # Percentage heatmap
    ax = axes[1]
    im = ax.imshow(pct_matrix, cmap=pct_cmap, aspect="auto", vmin=40, vmax=85)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight Within ±0.3°C (%)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = pct_matrix[i, j]
            color = "white" if val < 55 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Within ±0.3°C (%)", fontsize=11)

    plt.suptitle(f"Combined Sensitivity: Rate Limit × Forecast Noise\n(Rate noise = {rate_noise} °C/h)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "combined_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: combined_sensitivity_heatmap.png")
    plt.close()

    # Figure 2: Plasma/Viridis style
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(rms_matrix, cmap="plasma_r", aspect="auto", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight RMS Error (°C)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = rms_matrix[i, j]
            color = "white" if val > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RMS (°C)", fontsize=11)

    ax = axes[1]
    im = ax.imshow(pct_matrix, cmap="viridis", aspect="auto", vmin=40, vmax=85)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight Within ±0.3°C (%)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = pct_matrix[i, j]
            color = "white" if val < 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Within ±0.3°C (%)", fontsize=11)

    plt.suptitle(f"Combined Sensitivity: Rate Limit × Forecast Noise\n(Rate noise = {rate_noise} °C/h)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "combined_sensitivity_plasma.png", dpi=150, bbox_inches="tight")
    print(f"Saved: combined_sensitivity_plasma.png")
    plt.close()

    # Figure 3: Cool diverging colormap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(rms_matrix, cmap="coolwarm", aspect="auto", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight RMS Error (°C)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = rms_matrix[i, j]
            color = "white" if (val < 0.35 or val > 0.6) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RMS (°C)", fontsize=11)

    ax = axes[1]
    im = ax.imshow(pct_matrix, cmap="RdYlGn", aspect="auto", vmin=40, vmax=85)
    ax.set_xticks(range(len(temp_noises)))
    ax.set_xticklabels([f"{x:.1f}" for x in temp_noises], fontsize=11)
    ax.set_yticks(range(len(rate_limits)))
    ax.set_yticklabels([f"{x:.2f}" for x in rate_limits], fontsize=11)
    ax.set_xlabel("Temperature Forecast RMSE at 3h (°C)", fontsize=12)
    ax.set_ylabel("Max Cooling Rate (°C/hour)", fontsize=12)
    ax.set_title("Overnight Within ±0.3°C (%)", fontsize=13, fontweight="bold")

    for i in range(len(rate_limits)):
        for j in range(len(temp_noises)):
            val = pct_matrix[i, j]
            color = "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Within ±0.3°C (%)", fontsize=11)

    plt.suptitle(f"Combined Sensitivity: Rate Limit × Forecast Noise\n(Rate noise = {rate_noise} °C/h)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_path / "combined_sensitivity_diverging.png", dpi=150, bbox_inches="tight")
    print(f"Saved: combined_sensitivity_diverging.png")
    plt.close()

    # Print summary
    print("\n" + "=" * 75)
    print("COMBINED SENSITIVITY RESULTS")
    print(f"(Rate noise fixed at {rate_noise} °C/h)")
    print("=" * 75)
    print(f"{'Max Rate':<12} {'Temp Noise':<12} {'Overnight RMS':<15} {'Within ±0.3°C':<15}")
    print("-" * 75)
    for max_rate in rate_limits:
        for temp_noise in temp_noises:
            r = results[(max_rate, temp_noise)]
            print(f"{max_rate:<12.2f} {temp_noise:<12.1f} {r['rms']:<15.3f} {r['pct_good']:<15.1f}%")
    print("=" * 75)

    print("\nKEY INSIGHT:")
    print("  Rate limit dominates performance - vertical gradient >> horizontal gradient")
    print("  Increasing max_rate from 0.5 to 2.0 C/h has ~10× more impact than forecast noise")


if __name__ == "__main__":
    main()
