# CLAUDE.md - Project Context for Claude Code

## Project Overview

This project develops and evaluates thermal control algorithms for the Rubin Observatory M1M3 primary/tertiary mirror at Cerro Pachon, Chile (30.2444S, 70.7494W, 2700m elevation). The mirror has a large thermal mass and must be kept slightly cooler than ambient air to avoid "mirror seeing" -- convective plumes from a warm mirror that degrade image quality.

## Current State

The active control algorithm is a **three-phase system** implemented in `three_phase_control.py`:

- **Phase 1 (Daytime):** Fixed setpoint at predicted sunset temperature minus 0.3C cold bias. Dome is closed, HVAC maintains temperature.
- **Phase 2 (Pre-sunset transition):** Linear ramp from fixed to tracking. Best config uses T1=-2h (start), T2=0h (end at sunset).
- **Phase 3 (Overnight):** Weighted lookahead over next 0-3 hours with rate compensation (0.5 * tau * dT/dt) and 0.3C cold bias.

### Current Physical Parameters

| Parameter | Value | Variable |
|-----------|-------|----------|
| Mirror thermal time constant | 3.0 hours | `TAU` |
| Max setpoint change rate | 0.7 C/hour | `MAX_RATE` |
| Cold bias (target below ambient) | 0.3 C | `COLD_BIAS` |
| Simulation timestep | 15 minutes | `DT` |

### Latest Results (tau=3.0h, max_rate=0.7C/h)

- Sunset error: -0.08 +/- 0.11C (100% within +/-0.3C)
- Overnight RMS: 0.44C
- Overnight within +/-0.3C: ~59%
- Tested on 392 nights (odd-date test set)

## Key Files

### Simulation Code
- `three_phase_control.py` -- Main optimizer. Grid search over phase2 algorithms x T1 x T2. Train/test split: even dates train, odd dates test. Generates `three_phase_control_*_best.png` and `*_examples.png`.
- `three_phase_performance_plots.py` -- Generates histograms, 20-night grid (5x4), and 6-night detailed views. Outputs `three_phase_*_histograms.png`, `*_20_nights.png`, `*_detailed_nights.png`.
- `sun_utils.py` -- Astropy-based sunrise/sunset calculator for Cerro Pachon. Provides `get_sun_times_for_date(date, start_time)` returning (sunset_hours, sunrise_next_hours) in hours from dataset start. Uses LRU cache.

### Analysis
- `control_comparison_histogram.py` -- Compares three-phase control against two baselines (match ambient, fixed -0.75C bias).
- `mirror_rate_histogram.py` -- Distribution of mirror vs ambient temperature change rates.
- `three_day_plot.py` -- Continuous 3-day visualization with phase shading.
- `analyze_warm_tail.py` -- Studies mirror-warmer-than-ambient events.

### Data (not in git)
- `temp_history_all_dec2025_sunrise_sunset.csv` -- Primary dataset. Columns: `timestamp`, `y` (temperature in C). ~50 days of continuous ambient temperature at sub-hourly resolution from Cerro Pachon, Dec 2025 - Jan 2026.
- `sunset_data_preprocessed.pkl` -- Preprocessed sunset data cache.

## Plot Naming Convention

Plots with parameter-specific names use the format: `three_phase_{rate}rate_{tau}tau_{description}.png`

Example: `three_phase_07rate_3tau_20_nights.png` = max rate 0.7C/h, tau 3h, 20-night grid plot.

Plots without parameter tags in the name are from the previous default parameters (tau=2h, max_rate=2C/h).

## Thermal Model

All simulations use a first-order thermal response:
```
dT_mirror/dt = (T_setpoint - T_mirror) / tau
```
Discretized as: `T_mirror[i] = T_mirror[i-1] + (T_setpoint[i] - T_mirror[i-1]) / TAU * DT`

The setpoint is rate-limited: `|dT_setpoint/dt| <= MAX_RATE`.

## Design Constraints and Preferences

- **Slightly cold is always preferred over slightly warm.** A warm mirror causes mirror seeing; a cold mirror does not (within reason).
- The 0.3C cold bias is intentional -- we want the mirror ~0.3C below ambient.
- Error is defined as `T_mirror - (T_ambient - COLD_BIAS)`, so negative error = colder than target (acceptable), positive error = warmer than target (bad).
- Performance is evaluated from sunset to sunrise only (overnight observing window).

## Running Simulations

```bash
# Full optimization + best-config plots
python3 three_phase_control.py

# Detailed performance plots (histograms, 20 nights, 6 detailed nights)
python3 three_phase_performance_plots.py
```

Both scripts read the CSV data file from an absolute path. If moving to a different machine, update the path in the `pd.read_csv()` call.

## Dependencies

numpy, pandas, matplotlib, astropy, scipy
