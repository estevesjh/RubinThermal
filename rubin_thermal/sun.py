"""
Sun position calculations for Cerro Pachon using astropy.

Provides accurate UTC-based sunset/sunrise times for the Rubin Observatory.
All times are in UTC.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy import units as u
from functools import lru_cache

from .config import CONFIG

# Observatory location from config
_loc = CONFIG["location"]
LAT = _loc["lat"]
LON = _loc["lon"]
ELEVATION = _loc["elevation"]

# Create astropy location object
LOCATION = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=ELEVATION * u.m)


@lru_cache(maxsize=1000)
def find_sun_event(date_str: str, event: str = "sunset", target_alt: float = 0.0):
    """
    Find when sun crosses a target altitude.

    Parameters
    ----------
    date_str : str
        Date in format 'YYYY-MM-DD'
    event : str
        'sunset' or 'sunrise'
    target_alt : float
        Target altitude in degrees (0 = horizon, -6 = civil twilight, etc.)

    Returns
    -------
    pandas.Timestamp or None
        UTC time of the event
    """
    if event == "sunset":
        # Search from 20:00 to 02:00 UTC (covers all seasons at this longitude)
        base = Time(f"{date_str} 20:00:00", scale="utc")
        times = base + np.linspace(0, 6, 360) * u.hour  # 1-minute resolution
        direction = "down"
    else:  # sunrise
        # Search from 06:00 to 14:00 UTC
        base = Time(f"{date_str} 06:00:00", scale="utc")
        times = base + np.linspace(0, 8, 480) * u.hour
        direction = "up"

    altaz_frame = AltAz(obstime=times, location=LOCATION)
    sun_altaz = get_sun(times).transform_to(altaz_frame)
    alts = sun_altaz.alt.deg

    for i in range(len(alts) - 1):
        if direction == "down" and alts[i] > target_alt and alts[i + 1] <= target_alt:
            # Linear interpolation for precise time
            frac = (alts[i] - target_alt) / (alts[i] - alts[i + 1])
            event_time = times[i] + frac * (times[i + 1] - times[i])
            return pd.Timestamp(event_time.datetime).tz_localize(None)
        elif direction == "up" and alts[i] < target_alt and alts[i + 1] >= target_alt:
            frac = (target_alt - alts[i]) / (alts[i + 1] - alts[i])
            event_time = times[i] + frac * (times[i + 1] - times[i])
            return pd.Timestamp(event_time.datetime).tz_localize(None)

    return None


def get_sunset_utc(date) -> pd.Timestamp | None:
    """
    Get sunset time (sun at horizon) in UTC.

    Parameters
    ----------
    date : datetime.date or str
        The date

    Returns
    -------
    pandas.Timestamp or None
        UTC time of sunset
    """
    if hasattr(date, "strftime"):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    return find_sun_event(date_str, event="sunset", target_alt=0.0)


def get_sunrise_utc(date) -> pd.Timestamp | None:
    """
    Get sunrise time (sun at horizon) in UTC.

    Parameters
    ----------
    date : datetime.date or str
        The date

    Returns
    -------
    pandas.Timestamp or None
        UTC time of sunrise
    """
    if hasattr(date, "strftime"):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    return find_sun_event(date_str, event="sunrise", target_alt=0.0)


def get_sun_times_for_date(date, data_start_time) -> tuple[float | None, float | None]:
    """
    Get sunset and next-day sunrise times in hours from data start.

    Parameters
    ----------
    date : datetime.date
        The date for sunset
    data_start_time : pandas.Timestamp
        The first timestamp in the data (for computing hours offset)

    Returns
    -------
    tuple (sunset_hours, sunrise_next_hours)
        Both in hours from data_start_time, or (None, None) if not found
    """
    sunset_utc = get_sunset_utc(date)

    # Get next day's sunrise
    next_date = date + pd.Timedelta(days=1)
    sunrise_utc = get_sunrise_utc(next_date)

    if sunset_utc is None or sunrise_utc is None:
        return None, None

    sunset_hours = (sunset_utc - data_start_time).total_seconds() / 3600
    sunrise_hours = (sunrise_utc - data_start_time).total_seconds() / 3600

    return sunset_hours, sunrise_hours


def build_sun_events(dates: list, data_start_time) -> list[dict]:
    """
    Build a list of sun events for multiple dates.

    Parameters
    ----------
    dates : list
        List of dates (datetime.date objects)
    data_start_time : pandas.Timestamp
        The first timestamp in the data

    Returns
    -------
    list of dict
        Each dict contains 'date', 'sunset', 'sunrise', 'sunset_utc', 'sunrise_utc'
    """
    sun_events = []

    for date in dates:
        sunset_utc = get_sunset_utc(date)
        next_date = date + pd.Timedelta(days=1)
        sunrise_utc = get_sunrise_utc(next_date)

        if sunset_utc is None or sunrise_utc is None:
            continue

        sunset_hours = (sunset_utc - data_start_time).total_seconds() / 3600
        sunrise_hours = (sunrise_utc - data_start_time).total_seconds() / 3600

        sun_events.append({
            "date": date,
            "sunset": sunset_hours,
            "sunrise": sunrise_hours,
            "sunset_utc": sunset_utc,
            "sunrise_utc": sunrise_utc,
        })

    return sun_events


if __name__ == "__main__":
    # Test the module
    print("Testing sun.py")
    print("=" * 60)

    test_dates = ["2024-01-15", "2024-03-11", "2024-06-15", "2024-09-15", "2024-11-22"]

    print(f"\nSun times for Cerro Pachon ({LAT}, {LON}):")
    print("-" * 60)

    for date_str in test_dates:
        sunset = get_sunset_utc(date_str)
        sunrise = get_sunrise_utc(date_str)
        if sunset and sunrise:
            print(
                f"{date_str}: Sunrise {sunrise.strftime('%H:%M')} UTC, "
                f"Sunset {sunset.strftime('%H:%M')} UTC"
            )
