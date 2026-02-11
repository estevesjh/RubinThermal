#!/usr/bin/env python3
"""
Three-Phase Control System for Mirror Thermal Management
Uses accurate astropy-based sun calculations.

Phase 1: DAYTIME (sunrise to T1 hours before sunset)
  - Dome closed
  - Fixed setpoint based on predicted sunset temperature

Phase 2: PRE-SUNSET TRANSITION (T1 to T2 hours before sunset)
  - Dome still closed but starting to track
  - Intermediate algorithm to bring mirror to correct temp at sunset

Phase 3: OVERNIGHT (T2 hours before sunset to sunrise)
  - Dome open (after sunset)
  - Full lookahead tracking algorithm
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import sun utilities
from sun_utils import get_sun_times_for_date, LAT, LON

# Physical parameters
TAU = 3.0  # Mirror thermal time constant (hours)
MAX_RATE = 0.7  # Maximum setpoint change rate (°C/hour)
COLD_BIAS = 0.3  # Target: 0.3°C below ambient
DT = 0.25  # Timestep (hours) = 15 minutes

# Load data
print("Loading data...")
df = pd.read_csv('/Users/christopherstubbs/Desktop/projects/RubinTemperature/temp_history_all_dec2025_sunrise_sunset.csv',
                 parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600

def get_temperature(t_hours):
    return np.interp(t_hours, df['hours'].values, df['y'].values)

# Build database using sun times
print("Building day+night database...")
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

        # Get temperature at sunset
        T_sunset = get_temperature(sunset_hours)

        # Build time series (-8h to +12h from sunset)
        t_start = sunset_hours - 8
        t_end = min(sunset_hours + 12, sunrise_next_hours)
        sim_times = np.arange(t_start, t_end, DT)
        sim_temps = np.array([get_temperature(t) for t in sim_times])
        hours_from_sunset = sim_times - sunset_hours

        days.append({
            'date': date,
            'sunset_hours': sunset_hours,
            'T_sunset': T_sunset,
            'hours_from_sunset': hours_from_sunset,
            'temps': sim_temps,
        })
    except Exception as e:
        continue

print(f"Found {len(days)} days")
train_days = [d for d in days if d['date'].day % 2 == 0]
test_days = [d for d in days if d['date'].day % 2 == 1]
print(f"Train: {len(train_days)}, Test: {len(test_days)}")


def interpolate_temp(hours, temps, t):
    if t <= hours[0]: return temps[0]
    if t >= hours[-1]: return temps[-1]
    return np.interp(t, hours, temps)

def compute_rate(hours, temps, t, window=1.0):
    t_before = max(t - window/2, hours[0])
    t_after = min(t + window/2, hours[-1])
    if t_after <= t_before: return 0.0
    return (interpolate_temp(hours, temps, t_after) -
            interpolate_temp(hours, temps, t_before)) / (t_after - t_before)


def make_linear_kernel(n_points, T_max):
    times = np.linspace(0, T_max, n_points)
    weights = (T_max - times) / T_max
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return weights, times


def simulate_three_phase(day, T1, T2, phase1_func, phase2_func, phase3_func):
    """
    Simulate three-phase control.

    T1: start of phase 2 (hours before sunset, negative)
    T2: start of phase 3 (hours before sunset, negative or zero)
    """
    hours = day['hours_from_sunset']
    temps = day['temps']
    T_sunset = day['T_sunset']

    t_sim = hours.copy()
    n = len(t_sim)

    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = np.zeros(n)
    phase = np.zeros(n, dtype=int)

    # Initialize mirror at daytime setpoint
    T_mirror[0] = phase1_func(t_sim[0], temps[0], T_sunset, 0, hours, temps)

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb
        rate = compute_rate(hours, temps, t, 1.0)

        if t < T1:
            # Phase 1: Daytime
            T_setpoint[i] = phase1_func(t, T_amb, T_sunset, rate, hours, temps)
            phase[i] = 1
        elif t < T2:
            # Phase 2: Pre-sunset transition
            T_setpoint[i] = phase2_func(t, T_amb, T_sunset, rate, hours, temps, T1, T2)
            phase[i] = 2
        else:
            # Phase 3: Overnight
            T_setpoint[i] = phase3_func(t, T_amb, T_sunset, rate, hours, temps)
            phase[i] = 3

        # Rate limiting
        if i > 0:
            max_change = MAX_RATE * DT
            delta = T_setpoint[i] - T_setpoint[i-1]
            if abs(delta) > max_change:
                T_setpoint[i] = T_setpoint[i-1] + np.sign(delta) * max_change

        # Thermal dynamics
        if i > 0:
            dT = (T_setpoint[i] - T_mirror[i-1]) / TAU * DT
            T_mirror[i] = T_mirror[i-1] + dT

    T_target = T_ambient - COLD_BIAS
    errors = T_mirror - T_target

    sunset_idx = np.argmin(np.abs(t_sim))
    error_at_sunset = errors[sunset_idx]

    overnight_mask = t_sim >= 0
    rms_overnight = np.sqrt(np.mean(errors[overnight_mask]**2)) if np.any(overnight_mask) else np.nan

    return {
        't': t_sim,
        'T_ambient': T_ambient,
        'T_target': T_target,
        'T_setpoint': T_setpoint,
        'T_mirror': T_mirror,
        'errors': errors,
        'phase': phase,
        'error_at_sunset': error_at_sunset,
        'rms_overnight': rms_overnight,
    }


# ============================================================
# PHASE ALGORITHMS
# ============================================================

def phase1_fixed(t, T_amb, T_sunset, rate, hours, temps):
    """Fixed at predicted sunset temp - bias."""
    return T_sunset - COLD_BIAS

def phase2_ramp(t, T_amb, T_sunset, rate, hours, temps, T1, T2):
    """Linear ramp from fixed to tracking."""
    alpha = (t - T1) / (T2 - T1)
    fixed = T_sunset - COLD_BIAS
    tracking = T_amb - COLD_BIAS + 0.5 * TAU * rate
    return (1 - alpha) * fixed + alpha * tracking

def phase2_track_rate(t, T_amb, T_sunset, rate, hours, temps, T1, T2):
    """Track ambient with rate compensation."""
    return T_amb - COLD_BIAS + 0.5 * TAU * rate

def phase2_aggressive_track(t, T_amb, T_sunset, rate, hours, temps, T1, T2):
    """Aggressive tracking with full rate compensation."""
    return T_amb - COLD_BIAS + TAU * rate

def phase2_lookahead_ramp(t, T_amb, T_sunset, rate, hours, temps, T1, T2):
    """Gradually introduce lookahead."""
    alpha = (t - T1) / (T2 - T1)
    simple = T_amb - COLD_BIAS + 0.5 * TAU * rate
    weights, times = make_linear_kernel(13, 3.0)
    weighted_temp = sum(w * interpolate_temp(hours, temps, t + dt)
                       for w, dt in zip(weights, times))
    lookahead = weighted_temp + 0.5 * TAU * rate - COLD_BIAS
    return (1 - alpha) * simple + alpha * lookahead

def phase3_lookahead(t, T_amb, T_sunset, rate, hours, temps):
    """Linear weighted lookahead + rate."""
    weights, times = make_linear_kernel(13, 3.0)
    weighted_temp = sum(w * interpolate_temp(hours, temps, t + dt)
                       for w, dt in zip(weights, times))
    return weighted_temp + 0.5 * TAU * rate - COLD_BIAS


# ============================================================
# GRID SEARCH
# ============================================================
print("\n" + "="*70)
print("THREE-PHASE OPTIMIZATION")
print("="*70)

phase2_options = [
    ("Ramp fixed→track", phase2_ramp),
    ("Track + 0.5τ×rate", phase2_track_rate),
    ("Track + τ×rate", phase2_aggressive_track),
    ("Lookahead ramp", phase2_lookahead_ramp),
]

T1_options = [-6, -5, -4, -3, -2]
T2_options = [-2, -1, 0]

results = []

print(f"\nTesting {len(phase2_options)} phase2 algorithms × {len(T1_options)} T1 values × {len(T2_options)} T2 values")

for phase2_name, phase2_func in phase2_options:
    for T1 in T1_options:
        for T2 in T2_options:
            if T2 <= T1:
                continue

            sunset_errors = []
            overnight_rms = []

            for day in train_days:
                result = simulate_three_phase(day, T1, T2, phase1_fixed, phase2_func, phase3_lookahead)
                sunset_errors.append(result['error_at_sunset'])
                if not np.isnan(result['rms_overnight']):
                    overnight_rms.append(result['rms_overnight'])

            sunset_errors = np.array(sunset_errors)
            overnight_rms = np.array(overnight_rms)

            results.append({
                'phase2': phase2_name,
                'T1': T1,
                'T2': T2,
                'phase2_func': phase2_func,
                'sunset_mean': np.mean(sunset_errors),
                'sunset_rms': np.sqrt(np.mean(sunset_errors**2)),
                'overnight_rms': np.mean(overnight_rms),
                'pct_sunset_good': 100 * np.mean(np.abs(sunset_errors) < 0.3),
            })

results.sort(key=lambda x: x['sunset_rms'] + x['overnight_rms'])

print("\nTop 15 configurations (training set):")
print("-"*90)
print(f"{'Rank':<5} {'Phase2 Algorithm':<20} {'T1':>4} {'T2':>4} {'Sunset RMS':>11} {'Overnight':>10} {'Total':>8}")
print("-"*90)
for i, r in enumerate(results[:15]):
    total = r['sunset_rms'] + r['overnight_rms']
    print(f"{i+1:<5} {r['phase2']:<20} {r['T1']:>4}h {r['T2']:>4}h {r['sunset_rms']:>10.3f}°C {r['overnight_rms']:>9.3f}°C {total:>7.3f}")


# ============================================================
# TEST SET EVALUATION
# ============================================================
print("\n" + "="*70)
print("TEST SET EVALUATION (Top 5)")
print("="*70)

test_results = []
for r in results[:5]:
    sunset_errors = []
    overnight_rms = []
    all_errors = []

    for day in test_days:
        result = simulate_three_phase(day, r['T1'], r['T2'],
                                      phase1_fixed, r['phase2_func'], phase3_lookahead)
        sunset_errors.append(result['error_at_sunset'])
        if not np.isnan(result['rms_overnight']):
            overnight_rms.append(result['rms_overnight'])
        overnight_mask = result['t'] >= 0
        all_errors.extend(result['errors'][overnight_mask])

    sunset_errors = np.array(sunset_errors)
    all_errors = np.array(all_errors)

    test_results.append({
        **r,
        'test_sunset_mean': np.mean(sunset_errors),
        'test_sunset_rms': np.sqrt(np.mean(sunset_errors**2)),
        'test_overnight_rms': np.mean(overnight_rms),
        'test_pct_sunset_good': 100 * np.mean(np.abs(sunset_errors) < 0.3),
        'test_all_errors': all_errors,
        'test_sunset_errors': sunset_errors,
    })

    print(f"{r['phase2']:<20} T1={r['T1']}h, T2={r['T2']}h")
    print(f"  Sunset: {np.mean(sunset_errors):+.3f}±{np.std(sunset_errors):.3f}°C, "
          f"within ±0.3°C: {100*np.mean(np.abs(sunset_errors)<0.3):.0f}%")
    print(f"  Overnight RMS: {np.mean(overnight_rms):.3f}°C")
    print()

best = test_results[0]


# ============================================================
# VISUALIZATION
# ============================================================

# Figure 1: Example day
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

example_day = test_days[25]
result = simulate_three_phase(example_day, best['T1'], best['T2'],
                              phase1_fixed, best['phase2_func'], phase3_lookahead)

ax = axes[0, 0]
ax.plot(result['t'], result['T_ambient'], 'b-', linewidth=2, label='Ambient')
ax.plot(result['t'], result['T_target'], 'k--', linewidth=1.5, label='Target (Amb-0.3)')
ax.plot(result['t'], result['T_setpoint'], 'r-', linewidth=1.5, alpha=0.7, label='Setpoint')
ax.plot(result['t'], result['T_mirror'], 'g-', linewidth=2.5, label='Mirror')

ax.axvline(x=0, color='orange', linewidth=2, linestyle='-', label='Sunset')
ax.axvline(x=best['T1'], color='purple', linewidth=1.5, linestyle='--', label=f'T1={best["T1"]}h')
ax.axvline(x=best['T2'], color='cyan', linewidth=1.5, linestyle='--', label=f'T2={best["T2"]}h')

colors = ['yellow', 'orange', 'lightblue']
t = result['t']
for p in [1, 2, 3]:
    mask = result['phase'] == p
    if np.any(mask):
        ax.axvspan(t[mask][0], t[mask][-1], alpha=0.15, color=colors[p-1])

ax.set_xlabel('Hours from Sunset', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title(f'Three-Phase Control: {example_day["date"]}\n'
             f'Sunset error: {result["error_at_sunset"]:+.2f}°C',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 8)

ax = axes[0, 1]
ax.plot(result['t'], result['errors'], 'g-', linewidth=2)
ax.axhline(y=0, color='black', linewidth=2)
ax.axhline(y=0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axhline(y=-0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axvspan(-0.3, 0.3, alpha=0.1, color='green')
ax.axvline(x=0, color='orange', linewidth=2)
ax.axvline(x=best['T1'], color='purple', linewidth=1.5, linestyle='--')
ax.axvline(x=best['T2'], color='cyan', linewidth=1.5, linestyle='--')

ax.set_xlabel('Hours from Sunset', fontsize=11)
ax.set_ylabel('Error: T_mirror - T_ambient + 0.3 (°C)', fontsize=11)
ax.set_title('Control Error', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 8)
ax.set_ylim(-1, 1)

ax = axes[1, 0]
ax.hist(best['test_sunset_errors'], bins=30, alpha=0.7, color='steelblue',
        edgecolor='navy', density=True)
ax.axvline(x=0, color='black', linewidth=2)
ax.axvline(x=0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axvline(x=-0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axvspan(-0.3, 0.3, alpha=0.15, color='green')

ax.set_xlabel('Error at Sunset (°C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Distribution of Sunset Errors\n'
             f'Mean: {np.mean(best["test_sunset_errors"]):+.3f}°C, '
             f'±0.3°C: {best["test_pct_sunset_good"]:.0f}%',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(best['test_all_errors'], bins=50, alpha=0.7, color='green',
        edgecolor='darkgreen', density=True)
ax.axvline(x=0, color='black', linewidth=2)
ax.axvline(x=0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axvline(x=-0.3, color='gray', linewidth=1.5, linestyle='--')
ax.axvspan(-0.3, 0.3, alpha=0.15, color='green')

pct_good = 100 * np.mean(np.abs(best['test_all_errors']) < 0.3)
ax.set_xlabel('Overnight Error (°C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Distribution of Overnight Errors\n'
             f'RMS: {best["test_overnight_rms"]:.3f}°C, ±0.3°C: {pct_good:.0f}%',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.5, 1.5)

plt.suptitle(f'Best Three-Phase Control: {best["phase2"]}, T1={best["T1"]}h, T2={best["T2"]}h',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('three_phase_control_07rate_3tau_best.png', dpi=150, bbox_inches='tight')
print("\nSaved: three_phase_control_best.png")
plt.close()


# Figure 2: Multiple example nights
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx in range(9):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    day = test_days[idx * 40]
    result = simulate_three_phase(day, best['T1'], best['T2'],
                                  phase1_fixed, best['phase2_func'], phase3_lookahead)

    ax.plot(result['t'], result['T_ambient'], 'b-', linewidth=1.5, label='Ambient')
    ax.plot(result['t'], result['T_target'], 'k--', linewidth=1, label='Target')
    ax.plot(result['t'], result['T_mirror'], 'g-', linewidth=2, label='Mirror')

    ax.axvline(x=0, color='orange', linewidth=2)
    ax.axvline(x=best['T1'], color='purple', linewidth=1, linestyle='--')
    ax.axvline(x=best['T2'], color='cyan', linewidth=1, linestyle='--')

    for p, color in [(1, 'yellow'), (2, 'orange'), (3, 'lightblue')]:
        mask = result['phase'] == p
        if np.any(mask):
            ax.axvspan(result['t'][mask][0], result['t'][mask][-1], alpha=0.1, color=color)

    ax.set_title(f'{day["date"]}\nSunset err: {result["error_at_sunset"]:+.2f}°C, '
                 f'Night RMS: {result["rms_overnight"]:.2f}°C', fontsize=10)
    ax.set_xlabel('Hours from Sunset', fontsize=9)
    ax.set_ylabel('Temp (°C)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 8)

    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle(f'Three-Phase Control: Multiple Nights\n'
             f'{best["phase2"]}, T1={best["T1"]}h, T2={best["T2"]}h',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('three_phase_control_07rate_3tau_examples.png', dpi=150, bbox_inches='tight')
print("Saved: three_phase_control_examples.png")
plt.close()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("OPTIMAL THREE-PHASE CONTROL SYSTEM")
print("="*70)
print(f"""
PHASE 1: DAYTIME (t < {best['T1']}h before sunset)
  Algorithm: Fixed setpoint at T_predicted_sunset - 0.3°C

PHASE 2: PRE-SUNSET TRANSITION ({best['T1']}h ≤ t < {best['T2']}h)
  Algorithm: {best['phase2']}
  Duration: {abs(best['T2'] - best['T1'])} hours

PHASE 3: OVERNIGHT (t ≥ {best['T2']}h)
  Algorithm: Linear weighted lookahead (0-3h) + 0.5τ×rate - 0.3°C

PERFORMANCE (test set, {len(test_days)} nights):
  Sunset error: {np.mean(best['test_sunset_errors']):+.3f} ± {np.std(best['test_sunset_errors']):.3f}°C
  Sunset ±0.3°C: {best['test_pct_sunset_good']:.0f}%
  Overnight RMS: {best['test_overnight_rms']:.3f}°C
  Overnight ±0.3°C: {100*np.mean(np.abs(best['test_all_errors']) < 0.3):.0f}%
""")
