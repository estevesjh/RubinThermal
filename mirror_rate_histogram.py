#!/usr/bin/env python3
"""Histogram of M1M3 temperature rate of change during nighttime."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sun_utils import get_sun_times_for_date

# Load data
df = pd.read_csv('temp_history_all_dec2025_sunrise_sunset.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

def get_temperature(t_hours):
    return np.interp(t_hours, df['hours'].values, df['y'].values)

# Physical parameters
TAU = 2.0
MAX_RATE = 2.0
COLD_BIAS = 0.3
DT = 0.25  # hours = 15 minutes
T1 = -3
T2 = 0

# Build database
days = []
dates = sorted(df['timestamp'].dt.date.unique())
start_time = df['timestamp'].iloc[0]

for date in dates[1:-1]:
    try:
        sunset_hours, sunrise_next_hours = get_sun_times_for_date(date, start_time)
        if sunset_hours is None or sunrise_next_hours is None:
            continue
        if sunset_hours - 8 < df['hours'].iloc[0] or sunrise_next_hours > df['hours'].iloc[-1]:
            continue
        T_sunset = get_temperature(sunset_hours)
        t_start = sunset_hours - 8
        t_end = min(sunset_hours + 12, sunrise_next_hours)
        sim_times = np.arange(t_start, t_end, DT)
        sim_temps = np.array([get_temperature(t) for t in sim_times])
        hours_from_sunset = sim_times - sunset_hours
        days.append({
            'date': date, 'sunset_hours': sunset_hours, 'T_sunset': T_sunset,
            'hours_from_sunset': hours_from_sunset, 'temps': sim_temps,
        })
    except:
        continue

test_days = [d for d in days if d['date'].day % 2 == 1]
print(f"Analyzing {len(test_days)} test nights")

def interpolate_temp(hours, temps, t):
    if t <= hours[0]: return temps[0]
    if t >= hours[-1]: return temps[-1]
    return np.interp(t, hours, temps)

def compute_rate(hours, temps, t, window=1.0):
    t_before = max(t - window/2, hours[0])
    t_after = min(t + window/2, hours[-1])
    if t_after <= t_before: return 0.0
    return (interpolate_temp(hours, temps, t_after) - interpolate_temp(hours, temps, t_before)) / (t_after - t_before)

def make_linear_kernel(n_points, T_max):
    times = np.linspace(0, T_max, n_points)
    weights = (T_max - times) / T_max
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return weights, times

# Simulate and collect mirror temperature rates
all_mirror_rates = []
all_ambient_rates = []

for day in test_days:
    hours = day['hours_from_sunset']
    temps = day['temps']
    T_sunset = day['T_sunset']
    n = len(hours)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_mirror[0] = T_sunset - COLD_BIAS

    for i in range(n):
        t = hours[i]
        T_amb = temps[i]
        rate = compute_rate(hours, temps, t, 1.0)

        if t < T1:
            T_setpoint[i] = T_sunset - COLD_BIAS
        elif t < T2:
            alpha = (t - T1) / (T2 - T1)
            fixed = T_sunset - COLD_BIAS
            tracking = T_amb - COLD_BIAS + 0.5 * TAU * rate
            T_setpoint[i] = (1 - alpha) * fixed + alpha * tracking
        else:
            weights, times = make_linear_kernel(13, 3.0)
            weighted_temp = sum(w * interpolate_temp(hours, temps, t + dt) for w, dt in zip(weights, times))
            T_setpoint[i] = weighted_temp + 0.5 * TAU * rate - COLD_BIAS

        if i > 0:
            max_change = MAX_RATE * DT
            delta = T_setpoint[i] - T_setpoint[i-1]
            if abs(delta) > max_change:
                T_setpoint[i] = T_setpoint[i-1] + np.sign(delta) * max_change
            dT = (T_setpoint[i] - T_mirror[i-1]) / TAU * DT
            T_mirror[i] = T_mirror[i-1] + dT

    # Compute rates during overnight (t >= 0)
    overnight_mask = hours >= 0
    overnight_indices = np.where(overnight_mask)[0]

    for i in overnight_indices[1:]:  # Skip first point
        # Mirror rate of change (°C/hour)
        mirror_rate = (T_mirror[i] - T_mirror[i-1]) / DT
        all_mirror_rates.append(mirror_rate)

        # Ambient rate for comparison
        ambient_rate = (temps[i] - temps[i-1]) / DT
        all_ambient_rates.append(ambient_rate)

mirror_rates = np.array(all_mirror_rates)
ambient_rates = np.array(all_ambient_rates)

print(f"Total overnight rate samples: {len(mirror_rates)}")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Mirror rate histogram
ax = axes[0]
bins = np.linspace(-2, 2, 81)
ax.hist(mirror_rates, bins=bins, alpha=0.7, color='green', edgecolor='darkgreen', density=True)
ax.axvline(x=0, color='black', linewidth=2)
ax.axvline(x=np.mean(mirror_rates), color='red', linewidth=2, linestyle='--',
           label=f'Mean: {np.mean(mirror_rates):+.3f}°C/hr')

ax.set_xlabel('dT_mirror/dt (°C/hour)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('M1M3 Temperature Rate of Change (Overnight)\n'
             f'Std: {np.std(mirror_rates):.3f}°C/hr', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)

# Panel 2: Comparison with ambient rate
ax = axes[1]
ax.hist(ambient_rates, bins=bins, alpha=0.5, color='blue', edgecolor='navy',
        density=True, label=f'Ambient (std={np.std(ambient_rates):.3f}°C/hr)')
ax.hist(mirror_rates, bins=bins, alpha=0.5, color='green', edgecolor='darkgreen',
        density=True, label=f'Mirror (std={np.std(mirror_rates):.3f}°C/hr)')
ax.axvline(x=0, color='black', linewidth=2)

ax.set_xlabel('dT/dt (°C/hour)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Rate of Change: Mirror vs Ambient', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)

plt.tight_layout()
plt.savefig('mirror_rate_histogram.png', dpi=150, bbox_inches='tight')
print("\nSaved: mirror_rate_histogram.png")
plt.close()

# Print statistics
print("\n" + "="*60)
print("M1M3 TEMPERATURE RATE OF CHANGE STATISTICS (overnight)")
print("="*60)
print(f"Mean:   {np.mean(mirror_rates):+.4f} °C/hour")
print(f"Std:    {np.std(mirror_rates):.4f} °C/hour")
print(f"Min:    {np.min(mirror_rates):+.4f} °C/hour")
print(f"Max:    {np.max(mirror_rates):+.4f} °C/hour")
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p:2d}th: {np.percentile(mirror_rates, p):+.4f} °C/hour")

print(f"\n" + "="*60)
print("FRACTION OF TIME AT VARIOUS RATES:")
print("="*60)
for thresh in [0.5, 1.0, 1.5, 2.0]:
    pct_fast = 100 * np.mean(np.abs(mirror_rates) > thresh)
    print(f"  |rate| > {thresh:.1f}°C/hr: {pct_fast:.2f}%")
