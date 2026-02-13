"""
Configuration management for RubinThermal.

Loads parameters from config.yaml and provides access to project paths.
"""

import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (where config.yaml lives)."""
    # Start from this file and go up to find config.yaml
    current = Path(__file__).parent.parent
    if (current / "config.yaml").exists():
        return current
    # Fallback: check current working directory
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    raise FileNotFoundError(
        "Could not find config.yaml. Run from project root or set RUBIN_THERMAL_ROOT."
    )


def load_config(path: str | Path | None = None) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Path to config file. If None, uses config.yaml in project root.

    Returns
    -------
    dict
        Configuration dictionary with keys: physics, control, location, paths
    """
    if path is None:
        path = get_project_root() / "config.yaml"

    with open(path) as f:
        return yaml.safe_load(f)


# Load default config on import
try:
    CONFIG = load_config()
except FileNotFoundError:
    # Provide defaults if config.yaml not found (for testing)
    CONFIG = {
        "physics": {
            "tau": 3.0,
            "max_rate": 0.7,
            "cold_bias": 0.3,
            "dt": 0.25,
        },
        "control": {
            "t1": -3,
            "t2": 0,
            "lookahead_hours": 3.0,
            "lookahead_points": 13,
        },
        "location": {
            "lat": -30.2444,
            "lon": -70.7494,
            "elevation": 2700,
        },
        "paths": {
            "data": "data/temp_history_all_dec2025_sunrise_sunset.csv",
            "figures": "figures",
        },
    }


# Convenience accessors
def get_physics_params() -> dict:
    """Get physics parameters (tau, max_rate, cold_bias, dt)."""
    return CONFIG["physics"]


def get_control_params() -> dict:
    """Get control parameters (t1, t2, lookahead_hours, lookahead_points)."""
    return CONFIG["control"]


def get_location() -> dict:
    """Get observatory location (lat, lon, elevation)."""
    return CONFIG["location"]


def get_data_path() -> Path:
    """Get absolute path to temperature data file."""
    return get_project_root() / CONFIG["paths"]["data"]


def get_figures_path() -> Path:
    """Get absolute path to figures output directory."""
    return get_project_root() / CONFIG["paths"]["figures"]
