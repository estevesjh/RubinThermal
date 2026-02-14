# M1M3 Thermal Balance Study with Multi-Sensor Data

## Context

The current thermal control simulation uses a simplified two-zone model (surface/bulk) with synthetic gradient estimates. However, the real M1M3 mirror has **>30 temperature sensors** distributed across the mirror structure, providing:
- **Radial temperature gradients** (center to edge)
- **Z-axis gradients** (top/bottom of mirror)
- **Azimuthal variations** (honeycomb pattern effects)

This study aims to integrate real sensor data to validate and refine the thermal model, quantify actual gradient behavior, and improve control strategies.

---

## Data Requirements

### 1. Sensor Metadata File (`sensors_metadata.csv`)

Using simplified **(r, z) coordinate system** - radial distance and height only.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `sensor_id` | string | EFD topic/field name | `"lsst.sal.MTM1M3.thermalData.temperature0"` |
| `r` | float | Radial position (meters from center) | `0.0`, `1.5`, `3.0`, `4.2` |
| `z` | float | Z-axis position (+1=top, -1=bottom) | `1.0`, `-1.0` |
| `zone` | string | Logical zone assignment | `"center"`, `"inner"`, `"outer"` |
| `type` | string | Top or bottom | `"top"`, `"bottom"` |

**Example format:**
```csv
sensor_id,r,z,zone,type
lsst.sal.MTM1M3.thermalData.temperature0,0.0,1.0,center,top
lsst.sal.MTM1M3.thermalData.temperature1,0.0,-1.0,center,bottom
lsst.sal.MTM1M3.thermalData.temperature2,1.5,1.0,inner,top
lsst.sal.MTM1M3.thermalData.temperature3,1.5,-1.0,inner,bottom
lsst.sal.MTM1M3.thermalData.temperature4,3.0,1.0,middle,top
...
```

### 2. Time-Series Temperature Data (from EFD)

Data will be queried from the **Rubin Engineering Facilities Database (EFD)**.

**EFD Topics needed:**
```
lsst.sal.MTM1M3.thermalData          # M1M3 thermal sensors (~30 channels)
lsst.sal.ESS.temperature (salIndex=103)  # Dome interior temperature (primary reference)
lsst.sal.ESS.temperature (external)  # External ambient (when dome open)
lsst.sal.ATDome.position             # Dome open/closed status
lsst.sal.MTM1M3.appliedForces        # HVAC/thermal control status
```

**Query approach:**
```python
from lsst_efd_client import EfdClient

async def fetch_thermal_data(start_time, end_time):
    client = EfdClient('usdf_efd')  # or 'summit_efd'

    # Query M1M3 thermal sensors
    thermal_df = await client.select_time_series(
        'lsst.sal.MTM1M3.thermalData',
        ['temperature0', 'temperature1', ..., 'temperature29'],
        start_time, end_time
    )
    return thermal_df
```

**Export to CSV for offline analysis:**
```csv
timestamp,temperature0,temperature1,...,temperature29,ambient
2024-01-15 18:00:00,8.52,8.48,...,8.55,8.70
2024-01-15 18:15:00,8.45,8.42,...,8.48,8.62
```

### 3. Required Time Resolution

| Parameter | Minimum | Preferred |
|-----------|---------|-----------|
| Sampling interval | 15 min | 5 min |
| Duration per night | Sunset to sunrise | -2h to +10h from sunset |
| Number of nights | 10 | 30+ |

### 4. Ambient Reference Data

**Primary reference: Dome interior temperature (salIndex=103)**

When the dome is closed, the mirror should track the dome interior air temperature, not external ambient. This is critical for daytime thermal control.

| Dome State | Reference Sensor | salIndex |
|------------|------------------|----------|
| **Closed** | Dome interior temp | `salIndex=103` |
| **Open** | External ambient | TBD |

**Additional data:**
- Dome open/closed status (`lsst.sal.ATDome.position`)
- HVAC setpoint history (if available)
- External ambient (for comparison)

### 5. Operational Context (Optional but Valuable)

- Dome open/closed status
- HVAC active/inactive status
- Observing vs maintenance periods
- Any known thermal events (dome flush, etc.)

---

## Implementation Plan

### Phase 1: Data Infrastructure

**1.1 Create EFD client wrapper**
- New file: `rubin_thermal/efd_client.py`
- Functions:
  - `fetch_m1m3_thermal(start, end)` → DataFrame with all thermal sensors
  - `fetch_dome_temp(start, end, salIndex=103)` → Dome interior temperature
  - `fetch_dome_state(start, end)` → Dome open/closed status
  - `export_to_csv(df, path)` → Save for offline analysis

**1.2 Create multi-sensor data loader**
- New file: `rubin_thermal/multi_sensor.py`
- Functions:
  - `load_sensor_metadata(path)` → DataFrame with (r, z) positions
  - `load_multi_sensor_data(path, metadata_path)` → Dict with arrays
  - `build_multi_sensor_database(df, metadata)` → List of day dicts

**1.3 Extend day database structure**
```python
day = {
    "date": datetime.date,
    "sunset_hours": float,
    "hours_from_sunset": np.ndarray,      # Shape (n_times,)
    "temps_matrix": np.ndarray,           # Shape (n_times, n_sensors)
    "sensor_ids": list,                   # Length n_sensors
    "sensor_r": np.ndarray,               # Shape (n_sensors,) - radial position
    "sensor_z": np.ndarray,               # Shape (n_sensors,) - height (+1=top, -1=bot)
    "dome_temp": np.ndarray,              # Shape (n_times,) - salIndex=103
    "dome_open": np.ndarray,              # Shape (n_times,) - boolean, dome state
    "ambient_external": np.ndarray,       # Shape (n_times,) - external ambient
}
```

### Phase 2: Gradient Analysis Tools

**2.1 Radial gradient computation**
```python
def compute_radial_gradient(temps_matrix, sensor_positions):
    """
    Compute radial temperature gradient dT/dr at each timestep.
    Returns: gradient array shape (n_times,)
    """
```

**2.2 Z-axis gradient computation**
```python
def compute_z_gradient(temps_matrix, sensor_positions):
    """
    Compute top-bottom gradient (T_top - T_bottom) at each timestep.
    Returns: gradient array shape (n_times,)
    """
```

**2.3 Gradient statistics**
- Mean, std, percentiles for each gradient type
- Correlation with ambient rate of change
- Correlation with setpoint rate

### Phase 3: Multi-Zone Thermal Model

**3.1 Extend TwoZoneThermalModel to N-zones**
```python
class MultiZoneThermalModel:
    def __init__(self,
                 n_zones: int,
                 tau_zones: np.ndarray,
                 coupling_matrix: np.ndarray,
                 max_rate: float,
                 dt: float):
        ...
```

**3.2 Zone definitions for M1M3**
| Zone | Description | τ estimate | Sensors |
|------|-------------|------------|---------|
| `center_top` | Center, top surface | 1.0h | M1M3_T_001, ... |
| `center_bulk` | Center, bulk | 3.0h | M1M3_T_002, ... |
| `inner_top` | Inner ring, top | 1.5h | M1M3_T_003, ... |
| `inner_bulk` | Inner ring, bulk | 3.5h | M1M3_T_004, ... |
| `outer_top` | Outer ring, top | 2.0h | M1M3_T_005, ... |
| `outer_bulk` | Outer ring, bulk | 4.0h | M1M3_T_006, ... |

### Phase 4: Analysis Scripts

**4.1 `scripts/sensor_analysis.py`**
- Load multi-sensor data
- Compute radial and z-gradients over time
- Generate gradient histograms and time series plots
- Output: `figures/m1m3_gradient_analysis.png`

**4.2 `scripts/gradient_correlation.py`**
- Correlate measured gradients with:
  - Ambient temperature rate
  - Setpoint rate of change
  - Time since sunset
- Output: `figures/gradient_correlations.png`

**4.3 `scripts/zone_calibration.py`**
- Fit multi-zone model parameters to real data
- Estimate τ for each zone from step response
- Estimate coupling coefficients between zones
- Output: calibrated `config_multisensor.yaml`

### Phase 5: Validation & Reporting

**5.1 Compare simulated vs measured gradients**
- Run simulation with calibrated model
- Overlay measured sensor data
- Quantify model accuracy (RMS error per zone)

**5.2 Update safety analysis**
- Re-evaluate gradient thresholds with real data
- Identify worst-case sensor locations
- Refine rate limit recommendations

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `rubin_thermal/efd_client.py` | **CREATE** | EFD query wrapper for thermal data |
| `rubin_thermal/multi_sensor.py` | **CREATE** | Multi-sensor data loading and processing |
| `rubin_thermal/zones.py` | **CREATE** | Multi-zone thermal model |
| `scripts/fetch_efd_data.py` | **CREATE** | Script to query and export EFD data |
| `scripts/sensor_analysis.py` | **CREATE** | Gradient analysis from real data |
| `scripts/gradient_correlation.py` | **CREATE** | Correlation analysis |
| `scripts/zone_calibration.py` | **CREATE** | Parameter fitting |
| `config_multisensor.yaml` | **CREATE** | Multi-zone configuration |
| `data/sensors_metadata.csv` | **REQUIRED** | Sensor (r, z) positions |
| `data/m1m3_temps.csv` | **GENERATED** | Exported EFD data for offline analysis |

---

## Verification Plan

1. **Data loading test**: Load sample multi-sensor CSV, verify shape and structure
2. **Gradient computation test**: Verify radial/z-gradient calculation on synthetic data
3. **Model consistency test**: Run multi-zone model, verify energy conservation
4. **Regression test**: Compare single-zone results unchanged from current code

---

## Open Questions

1. **EFD access**: Which EFD instance to use - `usdf_efd` (USDF) or `summit_efd` (Cerro Pachon)?
2. **Sensor topic names**: Confirm the exact SAL topic path for M1M3 thermal sensors
3. **External ambient sensor**: Which salIndex for external ambient when dome is open?
4. **Data availability**: How many nights of commissioning data are available in the EFD?
5. **Sensor positions**: Do we have a document with exact (r, z) positions for each sensor?

**Resolved:**
- ✅ Dome interior temperature: `lsst.sal.ESS.temperature` with `salIndex=103`

---

## Next Steps

1. **Get EFD access credentials** and test connection to `usdf_efd` or `summit_efd`
2. **Identify sensor topic names** - query available M1M3 thermal topics
3. **Create `sensors_metadata.csv`** with (r, z) positions for each sensor
4. **Fetch sample data** - export 3-5 nights of thermal data to CSV
5. **Run initial gradient analysis** - compute radial and z-gradients
6. **Calibrate zone time constants** from measured thermal transients
