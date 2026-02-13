#!/usr/bin/env python3
"""
Generate a 3-day continuous simulation plot for three-phase thermal control.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rubin_thermal import (
    CONFIG,
    load_temperature_data,
    get_temperature,
    compute_rate,
    make_linear_kernel,
    get_sunset_utc,
    get_sunrise_utc,
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
    print("Loading data...")
    df = load_temperature_data()
    start_time = df["timestamp"].iloc[0]

    # Pick a 4-day simulation window
    base_date = df["timestamp"].iloc[0].date() + pd.Timedelta(days=30)
    plot_start_date = base_date + pd.Timedelta(days=1)
    print(f"Simulating 4 days starting from {base_date}, plotting final 3 days from {plot_start_date}")

    # Build continuous simulation over 4 days
    sim_start_hours = (pd.Timestamp(base_date) - start_time).total_seconds() / 3600
    sim_end_hours = sim_start_hours + 96  # 4 days
    plot_start_hours = sim_start_hours + 24  # Start plotting after 1 day

    # Get sun times for each day
    sun_events = []
    print("\nComputing sun times:")
    for day_offset in range(5):
        date = base_date + pd.Timedelta(days=day_offset)

        sunrise_utc = get_sunrise_utc(date)
        sunset_utc = get_sunset_utc(date)

        if sunrise_utc and sunset_utc:
            sunrise_hours = (sunrise_utc - start_time).total_seconds() / 3600
            sunset_hours = (sunset_utc - start_time).total_seconds() / 3600

            sun_events.append({
                "date": date,
                "sunrise": sunrise_hours,
                "sunset": sunset_hours,
                "sunrise_utc": sunrise_utc,
                "sunset_utc": sunset_utc,
            })
            print(f"  {date}: sunrise={sunrise_utc.strftime('%H:%M')} UTC, sunset={sunset_utc.strftime('%H:%M')} UTC")

    # Simulate
    t_sim = np.arange(sim_start_hours, sim_end_hours, model.dt)
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = np.zeros(n)
    phase = np.zeros(n, dtype=int)

    # Get all ambient temperatures first
    for i in range(n):
        T_ambient[i] = get_temperature(t_sim[i], df)

    # Initialize mirror at ambient - cold bias
    T_mirror[0] = T_ambient[0] - model.cold_bias

    for i in range(n):
        t = t_sim[i]
        T_amb = T_ambient[i]

        # Find relevant sunset and sunrise for this time
        relevant_sunset = None
        next_sunrise = None

        for j, evt in enumerate(sun_events):
            if t < evt["sunset"]:
                relevant_sunset = evt["sunset"]
                if j + 1 < len(sun_events):
                    next_sunrise = sun_events[j + 1]["sunrise"]
                else:
                    next_sunrise = evt["sunrise"] + 24
                break
            elif j + 1 < len(sun_events) and t < sun_events[j + 1]["sunrise"]:
                relevant_sunset = evt["sunset"]
                next_sunrise = sun_events[j + 1]["sunrise"]
                break

        if relevant_sunset is None:
            relevant_sunset = sun_events[-1]["sunset"]
            next_sunrise = sun_events[-1]["sunrise"] + 24

        hours_from_sunset = t - relevant_sunset
        is_nighttime = (t >= relevant_sunset) and (next_sunrise is not None) and (t < next_sunrise)

        # Compute rate using ambient data
        if i > 10:
            rate = compute_rate(t_sim[:i + 1], T_ambient[:i + 1], t, 1.0)
        else:
            rate = 0

        # Three-phase control
        if is_nighttime:
            # Phase 3: Overnight tracking
            weights, times = make_linear_kernel()
            weighted_temp = sum(w * get_temperature(t + dt, df) for w, dt in zip(weights, times))
            T_setpoint[i] = weighted_temp + 0.5 * model.tau * rate - model.cold_bias
            phase[i] = 3
        elif hours_from_sunset >= T1 and hours_from_sunset < T2:
            # Phase 2: Pre-sunset ramp
            alpha = (hours_from_sunset - T1) / (T2 - T1)
            T_sunset_expected = get_temperature(relevant_sunset, df)
            fixed = T_sunset_expected - model.cold_bias
            tracking = T_amb - model.cold_bias + 0.5 * model.tau * rate
            T_setpoint[i] = (1 - alpha) * fixed + alpha * tracking
            phase[i] = 2
        else:
            # Phase 1: Daytime fixed
            T_sunset_expected = get_temperature(relevant_sunset, df)
            T_setpoint[i] = T_sunset_expected - model.cold_bias
            phase[i] = 1

        # Rate limiting and thermal dynamics
        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    # Filter to only plot final 3 days
    plot_mask = t_sim >= plot_start_hours
    t_plot = t_sim[plot_mask] - plot_start_hours
    T_ambient_plot = T_ambient[plot_mask]
    T_mirror_plot = T_mirror[plot_mask]
    T_setpoint_plot = T_setpoint[plot_mask]
    phase_plot = phase[plot_mask]

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 8))

    # Phase shading at bottom
    phase_colors = {1: "yellow", 2: "orange", 3: "cornflowerblue"}
    current_phase = phase_plot[0]
    phase_start = 0
    for i in range(1, len(phase_plot)):
        if phase_plot[i] != current_phase or i == len(phase_plot) - 1:
            phase_end = t_plot[i] if phase_plot[i] != current_phase else t_plot[i]
            ax.axvspan(t_plot[phase_start], phase_end, ymin=0, ymax=0.08,
                       color=phase_colors[current_phase], alpha=0.8)
            current_phase = phase_plot[i]
            phase_start = i

    # Shade regions where mirror is above/below ambient
    ax.fill_between(t_plot, T_ambient_plot, T_mirror_plot,
                    where=(T_mirror_plot > T_ambient_plot),
                    color="red", alpha=0.4, label="Mirror > Ambient")
    ax.fill_between(t_plot, T_ambient_plot, T_mirror_plot,
                    where=(T_mirror_plot <= T_ambient_plot),
                    color="blue", alpha=0.4, label="Mirror < Ambient")

    # Plot temperatures
    ax.plot(t_plot, T_ambient_plot, "b-", linewidth=2, label="Ambient")
    ax.plot(t_plot, T_setpoint_plot, "r--", linewidth=1.5, alpha=0.8, label="Setpoint")
    ax.plot(t_plot, T_mirror_plot, "g-", linewidth=2.5, label="Mirror")

    # Mark sunrises and sunsets
    for evt in sun_events:
        t_sunrise = evt["sunrise"] - plot_start_hours
        t_sunset = evt["sunset"] - plot_start_hours
        if 0 <= t_sunrise <= 72:
            ax.axvline(x=t_sunrise, color="gold", linewidth=2, linestyle="-", alpha=0.8)
        if 0 <= t_sunset <= 72:
            ax.axvline(x=t_sunset, color="darkorange", linewidth=2, linestyle="-", alpha=0.8)

    # Shade nighttime periods
    for i, evt in enumerate(sun_events[:-1]):
        night_start = evt["sunset"] - plot_start_hours
        night_end = sun_events[i + 1]["sunrise"] - plot_start_hours
        if night_start < 72 and night_end > 0:
            ax.axvspan(max(0, night_start), min(72, night_end),
                       alpha=0.08, color="navy", zorder=0)

    # Day labels
    ylims = ax.get_ylim()
    y_top = ylims[1] if ylims[1] else 20
    for day_offset in range(3):
        noon_hours = day_offset * 24 + 12
        date = plot_start_date + pd.Timedelta(days=day_offset)
        ax.text(noon_hours, y_top - 0.3, date.strftime("%b %d"),
                ha="center", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Phase legend entries
    for p, (color, name) in [(1, ("yellow", "Phase 1: Daytime Fixed")),
                              (2, ("orange", "Phase 2: Pre-sunset Ramp")),
                              (3, ("cornflowerblue", "Phase 3: Overnight Track"))]:
        ax.fill_between([], [], color=color, alpha=0.8, label=name)

    ax.set_xlabel("Hours from Start", fontsize=12)
    ax.set_ylabel("Temperature (C)", fontsize=12)
    ax.set_title(
        "Three-Phase Thermal Control: 3-Day Simulation\n"
        "Phase bar at bottom | Orange line = Sunset | Gold line = Sunrise | Navy shading = Night",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2, bbox_to_anchor=(0.0, 0.95))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 72)
    ax.set_xticks(np.arange(0, 73, 6))

    plt.tight_layout()
    plt.savefig(figures_path / "three_phase_3day.png", dpi=150, bbox_inches="tight")
    print("\nSaved: three_phase_3day.png")
    plt.close()


if __name__ == "__main__":
    main()
