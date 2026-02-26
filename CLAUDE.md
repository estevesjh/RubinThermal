# CLAUDE.md - Project Context for Claude Code

## Project Overview

This project develops and evaluates thermal control algorithms for the Rubin Observatory M1M3 primary/tertiary mirror at Cerro Pachon, Chile (30.2444S, 70.7494W, 2700m elevation). The mirror has a large thermal mass and must be kept slightly cooler than ambient air to avoid "mirror seeing" -- convective plumes from a warm mirror that degrade image quality.

## Repository Structure

```
RubinThermal/
├── config.yaml              # All parameters in one place
├── rubin_thermal/           # Python package
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration loading
│   ├── data.py              # Data loading & database building
│   ├── physics.py           # Thermal model & simulation
│   ├── sun.py               # Sun position calculations
│   └── control.py           # Phase algorithms
│
├── scripts/                 # Thin scripts using the package
│   ├── optimize.py          # Grid search optimization
│   ├── performance.py       # Generate performance plots
│   ├── compare.py           # Compare control strategies
│   ├── rate_analysis.py     # Mirror rate histogram
│   ├── three_day.py         # 3-day continuous plot
│   └── warm_tail.py         # Warm tail analysis
│
├── tests/                   # Unit and integration tests (44 tests)
│   ├── conftest.py          # Pytest fixtures
│   ├── test_physics.py      # Physics functions tests
│   ├── test_data.py         # Data loading tests
│   ├── test_control.py      # Control algorithm tests
│   ├── test_simulation.py   # Integration tests
│   ├── fixtures/            # Test data samples
│   └── golden/              # Expected outputs for regression
│
├── data/                    # Temperature data
├── figures/                 # Output plots
├── docs/                    # Reports and presentations
├── notebooks/               # Jupyter notebooks
│
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Current State

The active control algorithm is a **three-phase system**:

- **Phase 1 (Daytime):** Fixed setpoint at predicted sunset temperature minus 0.3C cold bias. Dome is closed, HVAC maintains temperature.
- **Phase 2 (Pre-sunset transition):** Linear ramp from fixed to tracking. Best config uses T1=-2h (start), T2=0h (end at sunset).
- **Phase 3 (Overnight):** Weighted lookahead over next 0-3 hours with rate compensation (0.5 * tau * dT/dt) and 0.3C cold bias.

### Configuration

All parameters are centralized in `config.yaml`:

```yaml
physics:
  tau: 3.0              # Mirror thermal time constant (hours)
  max_rate: 0.7         # Maximum setpoint change rate (C/hour)
  cold_bias: 0.3        # Target below ambient (C)
  dt: 0.25              # Simulation timestep (hours)

control:
  t1: -2                # Phase 2 start (hours before sunset)
  t2: 0                 # Phase 3 start (at sunset)
  lookahead_hours: 3.0
  lookahead_points: 13
```

### Latest Results (tau=3.0h, max_rate=0.7C/h)

- Sunset error: -0.08 +/- 0.11C (100% within +/-0.3C)
- Overnight RMS: 0.44C
- Overnight within +/-0.3C: ~59%
- Tested on 392 nights (odd-date test set)

## Package Usage

```python
from rubin_thermal import (
    # Config
    CONFIG,
    load_config,
    get_project_root,
    # Data
    load_temperature_data,
    build_day_database,
    train_test_split,
    # Physics
    ThermalModel,
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    # Control
    three_phase_setpoint,
    # Sun
    get_sun_times_for_date,
)

# Load data
df = load_temperature_data()
days = build_day_database(df)
train_days, test_days = train_test_split(days)

# Create model with config parameters
model = ThermalModel()  # Uses CONFIG values by default

# Compute setpoint for a given time
setpoint, phase = three_phase_setpoint(t, T_amb, T_sunset, rate, hours, temps)

# Thermal dynamics step
new_temp = model.step(T_mirror_current, T_setpoint)
limited_setpoint = model.apply_rate_limit(new_setpoint, old_setpoint)
```

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

## Module Breakdown

### `rubin_thermal/config.py`
Loads `config.yaml` and provides `CONFIG` dict with all parameters.

### `rubin_thermal/data.py`
- `load_temperature_data()` -- Load CSV, add 'hours' column
- `build_day_database(df)` -- Build list of day dicts with sunset times and temp arrays
- `train_test_split(days)` -- Split by even/odd dates

### `rubin_thermal/physics.py`
- `interpolate_temp()`, `compute_rate()`, `make_linear_kernel()` -- Utility functions
- `thermal_step()` -- Single-step thermal dynamics
- `ThermalModel` -- Class encapsulating tau, max_rate, dt

### `rubin_thermal/control.py`
- `phase1_fixed()`, `phase2_ramp()`, `phase3_lookahead()` -- Phase algorithms
- `three_phase_setpoint()` -- Unified setpoint function returning (setpoint, phase)

### `rubin_thermal/sun.py`
- `get_sunset_utc()`, `get_sunrise_utc()` -- Sun position calculations using astropy
- `get_sun_times_for_date()` -- Get sunset/sunrise for a specific date

## Running Scripts

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization (grid search over T1, T2, phase2 algorithms)
python scripts/optimize.py

# Generate performance histograms and example night plots
python scripts/performance.py

# Compare three-phase vs baseline strategies
python scripts/compare.py

# Analyze mirror temperature change rates
python scripts/rate_analysis.py

# Generate 3-day continuous plot
python scripts/three_day.py

# Analyze warm tail events (mirror > ambient)
python scripts/warm_tail.py

# Run all tests (44 tests)
pytest tests/ -v
```

## Data Files

Temperature data in `data/`:
- `temp_history_all_dec2025_sunrise_sunset.csv` -- Primary dataset. Columns: `timestamp`, `y` (temperature in C). ~50 days of continuous ambient temperature at sub-hourly resolution from Cerro Pachon, Dec 2025 - Jan 2026.
- `temp_history_jan2026.csv` -- Additional January 2026 data.

## Plot Naming Convention

Plots with parameter-specific names use the format: `three_phase_{rate}rate_{tau}tau_{description}.png`

Example: `three_phase_07rate_3tau_20_nights.png` = max rate 0.7C/h, tau 3h, 20-night grid plot.

## Dependencies

See `requirements.txt`:
- numpy, pandas, matplotlib, scipy
- astropy (for sun position calculations)
- pyyaml (for configuration)
- pytest (for testing)

## EFD Temperature Query Scripts

Scripts for querying Rubin Observatory EFD (Engineering Facility Database) are in `/sdf/home/e/esteves/sitcom-analysis/queryTemps/`:

### Available Scripts

- `multi_sensor_query.py` -- Multi-sensor temperature query with mean/std aggregation
- `find_observing_nights.py` -- Identify nights when dome was open (from MTDome shutter position)
- `query_observing_nights.py` -- Query all sensors for identified observing nights (production pipeline)
- `query_m1m3_gradients.py` -- Query M1M3 thermocouple spatial gradients via ThermocoupleAnalysis
- `test_efd_setpoints_gradients.py` -- Test script for setpoints and gradient queries
- `add_dome_status.py` -- Augment existing data with dome aperture fraction
- `efd_temp_query.py` -- Original single-sensor query class
- `helper.py` -- File handling and twilight time utilities

### EFD Temperature Sensors

| Sensor | Topic | salIndex | Column(s) |
|--------|-------|----------|-----------|
| Outside temp | `lsst.sal.ESS.temperature` | 301 | `temperatureItem0` |
| Inside Top End Assembly | `lsst.sal.ESS.temperature` | 112 | `temperatureItem0` |
| Inside air (M1M3) | `lsst.sal.ESS.temperature` | 113 | `temperatureItem0` |
| Dome inside | `lsst.sal.ESS.temperature` | 111 | `temperatureItem0` |
| Wind | `lsst.sal.ESS.airFlow` | 301 | `speed`, `direction`, `maxSpeed` |

### M1M3 Thermal System (`lsst.sal.MTM1M3TS.glycolLoopTemperature`)

No salIndex needed - dedicated topic with columns:
- `aboveMirrorTemperature` -- Air above mirror
- `insideCellTemperature1/2/3` -- Three sensors inside cell (different locations)
- `mirrorCoolantSupplyTemperature` / `mirrorCoolantReturnTemperature`
- `telescopeCoolantSupplyTemperature` / `telescopeCoolantReturnTemperature`

### M1M3 Setpoint Commands (`lsst.sal.MTM1M3TS.command_applySetpoints`)

No salIndex needed. Commands issued every ~2.5 minutes during active control:
- `glycolSetpoint` -- Glycol loop target temperature (C)
- `heatersSetpoint` -- Heater target temperature (C), typically glycolSetpoint + 1.0C

### M1M3 Thermocouple Gradients

Queried via `lsst.ts.m1m3.utils.ThermocoupleAnalysis`, which internally queries `lsst.sal.ESS.temperature` with salIndex 114-117 (4 thermal scanners, 146 thermocouples total).

Gradients are computed by least-squares plane fitting to thermocouple positions:
- `x_gradient` / `y_gradient` -- East-West / North-South gradients (C/m)
- `z_gradient` -- Front-Back gradient (unitless, normalized z: 0=back, 1=front)
- `radial_gradient` -- Center-Edge gradient (C/m)

Operational limits: X/Y +/- 0.4C (across 8.4m), Z/Radial +/- 0.1C.

### Dome Status

Use `lsst.sal.MTDome.apertureShutter` with `positionActual0` and `positionActual1`. Dome is open when both shutters are in [90, 110] degrees.

### Query Usage

```bash
# Explore available columns
python multi_sensor_query.py --explore --start 2024-12-01 --end 2024-12-05

# Query specific sensors (including setpoints)
python multi_sensor_query.py --query --start 2024-12-01 --end 2024-12-15 \
    --sensors inside_air_m2 m1m3_glycol m1m3_setpoints outside_temp

# Find observing nights
python find_observing_nights.py --start 2024-01-01 --output observing_nights.csv

# Query all sensors for observing nights (1-min resolution, mean/std)
python query_observing_nights.py \
    --nights-csv /sdf/data/rubin/user/esteves/thermal_analysis/observing_nights_2025.csv \
    --start-night 2025-07-01 \
    --output-dir /sdf/data/rubin/user/esteves/thermal_analysis

# Query thermocouple gradients for observing nights (1-min resolution)
python query_m1m3_gradients.py \
    --nights-csv /sdf/data/rubin/user/esteves/thermal_analysis/observing_nights_2025.csv \
    --min-date 2025-07-01 --time-bin 60 \
    -o /sdf/data/rubin/user/esteves/thermal_analysis/gradient_data_observing_nights.csv

# Test setpoints and gradient queries
python test_efd_setpoints_gradients.py --all --hours 2
python test_efd_setpoints_gradients.py --explore  # Discover all setpoint columns
```

### Pre-Built Datasets

Thermal data in `/sdf/data/rubin/user/esteves/thermal_analysis/`:

| File | Description |
|------|-------------|
| `observing_nights_2025.csv` | 366 nights (Jan 2025 - Feb 2026) with dome open stats, 224 observing |
| `thermal_data_all_observing_nights.csv` | 173,084 rows, 166 nights (Jul 2025 - Feb 2026), 1-min, 37 columns |
| `gradient_data_observing_nights.csv` | 120,880 rows, 164 nights (Jul 2025 - Feb 2026), 1-min, 11 columns |

**Columns in `thermal_data_all_observing_nights.csv` (37 columns):**

| Category | Columns |
|----------|---------|
| Timestamp | `timestamp` (index, UTC) |
| Outside (ESS 301) | `outside_temp_mean/std` |
| Wind (ESS 301) | `wind_speed_mean/std`, `wind_direction_mean/std`, `wind_maxSpeed_mean/std` |
| Inside Top End Assembly (ESS 112) | `inside_m2_temp_mean/std` |
| Inside M1M3 (ESS 113) | `inside_m1m3_temp_mean/std` |
| M1M3 Glycol | `aboveMirrorTemperature_mean/std`, `insideCellTemperature1/2/3_mean/std`, `mirrorCoolantSupply/ReturnTemperature_mean/std`, `telescopeCoolantSupply/ReturnTemperature_mean/std` |
| M1M3 Setpoints | `glycolSetpoint_mean/std`, `heatersSetpoint_mean/std` |
| Dome Status | `positionActual0`, `positionActual1`, `sunAltitude`, `dome_open` |
| Metadata | `night_date` |

**Columns in `gradient_data_observing_nights.csv` (11 columns):**

| Category | Columns |
|----------|---------|
| Timestamp | `timestamp` (index, UTC) |
| Fit | `intercept`, `intercept_err` |
| Gradients | `x_gradient`, `y_gradient`, `z_gradient`, `radial_gradient` |
| Uncertainties | `x_gradient_err`, `y_gradient_err`, `z_gradient_err`, `radial_gradient_err` |
| Metadata | `night_date` |

**Data notes:**
- Thermocouple gradient data available from Jul 2025 onward (2 nights with gaps)
- Setpoint commands available from Jul 2025 onward (issued every ~2.5 min during active control)
- Night coverage: 6h before sunset to sunrise (UTC), night_date is Chile local evening date
