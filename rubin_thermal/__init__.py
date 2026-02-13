"""
RubinThermal - Mirror thermal control simulation package for Rubin Observatory.

This package provides thermal control algorithms for the M1M3 primary/tertiary
mirror at Cerro Pachon, Chile.
"""

from .config import CONFIG, load_config, get_project_root
from .data import load_temperature_data, get_temperature, build_day_database, train_test_split
from .physics import (
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    thermal_step,
    ThermalModel,
)
from .control import (
    phase1_fixed,
    phase2_ramp,
    phase2_track_rate,
    phase2_aggressive_track,
    phase2_lookahead_ramp,
    phase3_lookahead,
    three_phase_setpoint,
)
from .sun import (
    get_sunset_utc,
    get_sunrise_utc,
    get_sun_times_for_date,
    build_sun_events,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "CONFIG",
    "load_config",
    "get_project_root",
    # Data
    "load_temperature_data",
    "get_temperature",
    "build_day_database",
    "train_test_split",
    # Physics
    "interpolate_temp",
    "compute_rate",
    "make_linear_kernel",
    "thermal_step",
    "ThermalModel",
    # Control
    "phase1_fixed",
    "phase2_ramp",
    "phase2_track_rate",
    "phase2_aggressive_track",
    "phase2_lookahead_ramp",
    "phase3_lookahead",
    "three_phase_setpoint",
    # Sun
    "get_sunset_utc",
    "get_sunrise_utc",
    "get_sun_times_for_date",
    "build_sun_events",
]
