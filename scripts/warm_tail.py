#!/usr/bin/env python3
"""
Analyze how often the mirror exceeds ambient temperature.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

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


def main():
    model = ThermalModel()
    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    # Load data
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)

    # Simulate all test nights
    all_errors_vs_ambient = []

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

        # Get overnight errors (mirror - ambient)
        overnight_mask = hours >= 0
        errors_vs_ambient = T_mirror[overnight_mask] - temps[overnight_mask]
        all_errors_vs_ambient.extend(errors_vs_ambient)

    errors = np.array(all_errors_vs_ambient)

    print("=" * 60)
    print("MIRROR TEMPERATURE RELATIVE TO AMBIENT (overnight)")
    print("=" * 60)
    print(f"Total samples: {len(errors)}")
    print(f"\nMean: {np.mean(errors):+.3f}C")
    print(f"Std:  {np.std(errors):.3f}C")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {np.percentile(errors, p):+.3f}C")

    print(f"\n" + "=" * 60)
    print("FRACTION OF TIME MIRROR EXCEEDS AMBIENT BY THRESHOLD:")
    print("=" * 60)
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pct = 100 * np.mean(errors > thresh)
        print(f"  > {thresh:+.1f}C: {pct:5.2f}%")

    print(f"\n" + "=" * 60)
    print("FRACTION OF TIME MIRROR IS BELOW AMBIENT BY THRESHOLD:")
    print("=" * 60)
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        pct = 100 * np.mean(errors < -thresh)
        print(f"  < {-thresh:+.1f}C: {pct:5.2f}%")


if __name__ == "__main__":
    main()
