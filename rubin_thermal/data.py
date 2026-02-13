"""
Data loading and preprocessing for RubinThermal.

Consolidates CSV loading, temperature interpolation, and database building
that was previously duplicated across scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import CONFIG, get_data_path
from .sun import get_sun_times_for_date


def load_temperature_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and preprocess temperature data from CSV.

    Parameters
    ----------
    path : str or Path, optional
        Path to CSV file. If None, uses path from config.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, y (temperature), hours (from start)
    """
    if path is None:
        path = get_data_path()

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hours"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600

    return df


def get_temperature(t_hours: float, df: pd.DataFrame) -> float:
    """
    Get temperature at a given time by linear interpolation.

    Parameters
    ----------
    t_hours : float
        Time in hours from dataset start
    df : pd.DataFrame
        Temperature data with 'hours' and 'y' columns

    Returns
    -------
    float
        Interpolated temperature
    """
    return np.interp(t_hours, df["hours"].values, df["y"].values)


def build_day_database(df: pd.DataFrame, dt: float | None = None) -> list[dict]:
    """
    Build database of day/night simulation windows.

    Each entry contains temperature data centered around sunset, from
    8 hours before to sunrise (or 12 hours after, whichever is earlier).

    Parameters
    ----------
    df : pd.DataFrame
        Temperature data from load_temperature_data()
    dt : float, optional
        Timestep in hours. If None, uses config value.

    Returns
    -------
    list of dict
        Each dict contains:
        - date: datetime.date
        - sunset_hours: hours from data start to sunset
        - T_sunset: temperature at sunset
        - hours_from_sunset: array of times relative to sunset
        - temps: array of temperatures at those times
    """
    if dt is None:
        dt = CONFIG["physics"]["dt"]

    days = []
    dates = sorted(df["timestamp"].dt.date.unique())
    start_time = df["timestamp"].iloc[0]

    for date in dates[1:-1]:  # Skip first and last dates (incomplete data)
        try:
            sunset_hours, sunrise_next_hours = get_sun_times_for_date(date, start_time)

            if sunset_hours is None or sunrise_next_hours is None:
                continue

            # Check data coverage
            if sunset_hours - 8 < df["hours"].iloc[0]:
                continue
            if sunrise_next_hours > df["hours"].iloc[-1]:
                continue

            # Get temperature at sunset
            T_sunset = get_temperature(sunset_hours, df)

            # Build time series (-8h to +12h from sunset, or sunrise)
            t_start = sunset_hours - 8
            t_end = min(sunset_hours + 12, sunrise_next_hours)
            sim_times = np.arange(t_start, t_end, dt)
            sim_temps = np.array([get_temperature(t, df) for t in sim_times])
            hours_from_sunset = sim_times - sunset_hours

            days.append({
                "date": date,
                "sunset_hours": sunset_hours,
                "T_sunset": T_sunset,
                "hours_from_sunset": hours_from_sunset,
                "temps": sim_temps,
            })
        except Exception:
            continue

    return days


def train_test_split(days: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split days into train/test sets by even/odd date.

    Parameters
    ----------
    days : list of dict
        Output from build_day_database()

    Returns
    -------
    tuple (train_days, test_days)
        Train set contains even-dated days, test set contains odd-dated days.
    """
    train_days = [d for d in days if d["date"].day % 2 == 0]
    test_days = [d for d in days if d["date"].day % 2 == 1]
    return train_days, test_days
