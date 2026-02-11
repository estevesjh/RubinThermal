# Rubin Observatory Mirror Thermal Control

Simulation and optimization of the thermal control system for the Rubin Observatory M1M3 primary/tertiary mirror at Cerro Pachon, Chile.

## Problem

The M1M3 mirror has a large thermal time constant (~3 hours), meaning it cannot quickly track changes in ambient temperature. If the mirror is warmer than the surrounding air, convective plumes degrade image quality ("mirror seeing"). The goal is to keep the mirror approximately 0.3C below ambient at all times during nighttime observing, while respecting physical constraints on how fast the thermal control system can change its setpoint.

## Three-Phase Control Algorithm

The core approach divides each 24-hour cycle into three phases:

**Phase 1 -- Daytime** (until T1 hours before sunset)
- Dome is closed; HVAC maintains a fixed setpoint equal to the predicted sunset temperature minus a 0.3C cold bias.

**Phase 2 -- Pre-Sunset Transition** (T1 to T2 hours before sunset)
- Linear ramp from the fixed daytime setpoint toward active ambient tracking with rate compensation.
- Bridges the gap between the static daytime target and the dynamic overnight algorithm.

**Phase 3 -- Overnight** (T2 hours before sunset through sunrise)
- Weighted lookahead algorithm using predicted ambient temperatures over the next 0-3 hours.
- Includes rate-of-change compensation (0.5 * tau * dT/dt) to account for thermal lag.
- Targets mirror temperature = ambient - 0.3C.

### Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| tau | 3.0 h | Mirror thermal time constant |
| Max rate | 0.7 C/h | Maximum setpoint change rate |
| Cold bias | 0.3 C | Target offset below ambient |
| dt | 15 min | Simulation timestep |

## Repository Structure

### Simulation Code

- `three_phase_control.py` -- Main optimization script. Performs grid search over phase2 algorithms, T1, and T2 values using a train/test split (even/odd dates). Generates best-configuration plots.
- `three_phase_performance_plots.py` -- Generates detailed performance figures: error histograms, CDFs, 20-night overview grid, and 6-night detailed views.
- `sun_utils.py` -- Accurate sunrise/sunset calculations for Cerro Pachon using astropy. Provides `get_sun_times_for_date()` used by all simulation scripts.

### Analysis & Visualization

- `control_comparison_histogram.py` -- Compares three strategies: (1) match ambient, (2) fixed bias (ambient - 0.75C), (3) three-phase control. Generates overlay histograms.
- `mirror_rate_histogram.py` -- Analyzes distribution of mirror vs ambient temperature rates of change during nighttime.
- `three_day_plot.py` -- Generates a continuous 3-day simulation showing phase transitions, day/night cycles, and control behavior.
- `analyze_warm_tail.py` -- Studies the warm-side tail of the error distribution (mirror warmer than ambient).

### Reports

- `thermal_control_report.tex` -- Comprehensive thermal control methodology and results.
- `report1_prediction_methods.tex` -- Comparison of temperature prediction approaches.
- `report2_thermal_control.tex` -- Thermal control system analysis.
- `RAMP_algorithm_report.tex` -- Technical description of the RAMP control algorithm.
- `slides_thermal_summary.tex` -- Beamer presentation slides.
- `sunset_temperature_prediction_report.tex` -- Sunset temperature prediction methodology.

### Data

The temperature history CSV (not tracked in git due to size) contains ~50 days of continuous ambient temperature measurements from Cerro Pachon (December 2025 - January 2026) at sub-hourly resolution. Contact the authors for access.

## Usage

Run the optimization:
```bash
python3 three_phase_control.py
```

Generate performance plots:
```bash
python3 three_phase_performance_plots.py
```

Both scripts expect the data file `temp_history_all_dec2025_sunrise_sunset.csv` in the working directory.

### Dependencies

- numpy
- pandas
- matplotlib
- astropy
- scipy

## Results

With tau = 3.0 h and max setpoint rate = 0.7 C/h, the optimal configuration (Ramp fixed-to-track, T1 = -2h, T2 = 0h) achieves on the 392-night test set:

- **Sunset error:** -0.08 +/- 0.11 C (100% within +/-0.3C)
- **Overnight RMS:** 0.44 C
- **Overnight within +/-0.3C:** ~59%

## Location

Cerro Pachon, Chile: 30.2444 S, 70.7494 W, elevation 2700 m.
