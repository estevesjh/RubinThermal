#!/usr/bin/env python3
"""Analyze how often the mirror exceeds ambient temperature."""

import numpy as np
import pandas as pd
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
DT = 0.25
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

# Simulate all test nights
all_errors_vs_ambient = []

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

    # Get overnight errors (mirror - ambient)
    overnight_mask = hours >= 0
    errors_vs_ambient = T_mirror[overnight_mask] - temps[overnight_mask]
    all_errors_vs_ambient.extend(errors_vs_ambient)

errors = np.array(all_errors_vs_ambient)

print("="*60)
print("MIRROR TEMPERATURE RELATIVE TO AMBIENT (overnight)")
print("="*60)
print(f"Total samples: {len(errors)}")
print(f"\nMean: {np.mean(errors):+.3f}°C")
print(f"Std:  {np.std(errors):.3f}°C")
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p:2d}th: {np.percentile(errors, p):+.3f}°C")

print(f"\n" + "="*60)
print("FRACTION OF TIME MIRROR EXCEEDS AMBIENT BY THRESHOLD:")
print("="*60)
for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    pct = 100 * np.mean(errors > thresh)
    print(f"  > {thresh:+.1f}°C: {pct:5.2f}%")

print(f"\n" + "="*60)
print("FRACTION OF TIME MIRROR IS BELOW AMBIENT BY THRESHOLD:")
print("="*60)
for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    pct = 100 * np.mean(errors < -thresh)
    print(f"  < {-thresh:+.1f}°C: {pct:5.2f}%")
