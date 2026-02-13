# Rubin Observatory Mirror Thermal Control

Simulation and optimization of thermal control algorithms for the Rubin Observatory M1M3 primary/tertiary mirror at Cerro Pachon, Chile.

## The Problem

The M1M3 mirror has a large thermal mass (time constant ~3 hours). If the mirror is warmer than ambient air, convective plumes cause "mirror seeing" that degrades image quality. The goal is to keep the mirror **~0.3°C below ambient** during nighttime observing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization to find best parameters
python scripts/optimize.py

# Generate performance plots
python scripts/performance.py
```

## Repository Structure

```
RubinThermal/
├── config.yaml           # All tunable parameters
├── rubin_thermal/        # Python package
├── scripts/              # Analysis scripts
├── tests/                # Unit tests (44 tests)
├── data/                 # Temperature CSV files
├── figures/              # Output plots
├── docs/                 # Reports (PDF/TeX)
└── notebooks/            # Jupyter notebooks
```

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/optimize.py` | Grid search over T1, T2, algorithms | Best config + `*_best.png`, `*_examples.png` |
| `scripts/performance.py` | Evaluate current config | `*_histograms.png`, `*_20_nights.png`, `*_detailed_nights.png` |
| `scripts/compare.py` | Compare vs baseline strategies | `control_comparison_*.png` |
| `scripts/rate_analysis.py` | Mirror rate-of-change analysis | `mirror_rate_histogram.png` |
| `scripts/three_day.py` | 3-day continuous simulation | `three_phase_3day.png` |
| `scripts/warm_tail.py` | Analyze warm events (mirror > ambient) | Console output |

### Recommended Order

1. **`optimize.py`** — Find optimal T1, T2, and phase2 algorithm
2. **`performance.py`** — Generate detailed performance plots
3. **`compare.py`** — See improvement over baselines

## Output Plots

All plots are saved to `figures/`. Key outputs:

| Plot | Description |
|------|-------------|
| `three_phase_07rate_3tau_histograms.png` | Error distribution histograms + CDF |
| `three_phase_07rate_3tau_20_nights.png` | Grid of 20 example nights |
| `three_phase_07rate_3tau_detailed_nights.png` | 6 nights with error traces |
| `control_comparison_histogram.png` | Three-phase vs baselines |
| `three_phase_3day.png` | 3-day continuous simulation |

## Configuration

All parameters are in `config.yaml`:

```yaml
physics:
  tau: 3.0              # Mirror thermal time constant (hours)
  max_rate: 0.7         # Maximum setpoint change rate (C/hour)
  cold_bias: 0.3        # Target below ambient (C)

control:
  t1: -2                # Phase 2 start (hours before sunset)
  t2: 0                 # Phase 3 start (at sunset)
```

## The Three-Phase Algorithm

| Phase | Time | Algorithm |
|-------|------|-----------|
| **1. Daytime** | t < T1 | Fixed setpoint = predicted sunset temp - 0.3°C |
| **2. Transition** | T1 ≤ t < T2 | Linear ramp from fixed → tracking |
| **3. Overnight** | t ≥ T2 | Weighted lookahead + rate compensation |

## Results

Optimal config: **T1=-2h, T2=0h, Ramp fixed→track**

| Metric | Value |
|--------|-------|
| Sunset error | -0.08 ± 0.11°C |
| Sunset within ±0.3°C | 100% |
| Overnight RMS | 0.44°C |
| Overnight within ±0.3°C | ~59% |

*Tested on 392 nights (odd-date test set)*

## Data

Temperature data goes in `data/`:
- `temp_history_all_dec2025_sunrise_sunset.csv` — Primary dataset (~50 days)
- `temp_history_jan2026.csv` — Additional data

## Reports

Technical documentation in `docs/`:
- `thermal_control_report.pdf` — Main methodology
- `RAMP_algorithm_report.pdf` — Algorithm details
- `slides_thermal_summary.pdf` — Presentation slides

## Running Tests

```bash
pytest tests/ -v
```

44 tests covering physics, data loading, control algorithms, and simulation.

## Location

Cerro Pachon, Chile: 30.2444°S, 70.7494°W, 2700m elevation

## Dependencies

- numpy, pandas, matplotlib, scipy
- astropy (sun calculations)
- pyyaml (configuration)
- pytest (testing)
