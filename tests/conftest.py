"""
Pytest fixtures for RubinThermal tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_time_series():
    """Simple temperature time series for testing."""
    hours = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    temps = np.array([15.0, 14.0, 13.0, 12.0, 11.5, 11.0])
    return hours, temps


@pytest.fixture
def sample_day():
    """Sample day dictionary for simulation testing."""
    hours_from_sunset = np.arange(-6, 8, 0.25)
    n = len(hours_from_sunset)

    # Simulate a typical sunset temperature profile
    # Warm during day, cooling after sunset
    temps = np.zeros(n)
    for i, t in enumerate(hours_from_sunset):
        if t < -2:
            temps[i] = 18.0 + 0.5 * np.sin(t * 0.5)  # Warm daytime
        elif t < 0:
            temps[i] = 17.0 - 0.5 * (t + 2)  # Cooling before sunset
        else:
            temps[i] = 16.0 - 0.5 * t + 0.2 * np.sin(t)  # Cooling overnight

    return {
        "date": pd.Timestamp("2024-01-15").date(),
        "sunset_hours": 100.0,  # Arbitrary
        "T_sunset": 16.0,
        "hours_from_sunset": hours_from_sunset,
        "temps": temps,
    }


@pytest.fixture
def sample_dataframe():
    """Sample temperature DataFrame for data loading tests."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    temps = 15.0 + 5.0 * np.sin(np.arange(100) * 2 * np.pi / 24)  # Daily cycle

    df = pd.DataFrame({
        "timestamp": dates,
        "y": temps,
    })
    df["hours"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600
    return df


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent
