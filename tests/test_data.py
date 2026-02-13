"""
Unit tests for rubin_thermal.data module.
"""

import pytest
import numpy as np
import pandas as pd


class TestGetTemperature:
    """Tests for get_temperature function."""

    def test_interpolation(self, sample_dataframe):
        """Temperature interpolation works correctly."""
        from rubin_thermal.data import get_temperature

        df = sample_dataframe

        # Test at exact point
        t = 0.0
        expected = df.loc[0, "y"]
        result = get_temperature(t, df)
        assert abs(result - expected) < 1e-10

    def test_midpoint_interpolation(self, sample_dataframe):
        """Interpolates correctly between points."""
        from rubin_thermal.data import get_temperature

        df = sample_dataframe

        # Midpoint between first two values
        t = 0.5
        expected = (df.loc[0, "y"] + df.loc[1, "y"]) / 2
        result = get_temperature(t, df)
        assert abs(result - expected) < 0.1  # Allow small tolerance


class TestTrainTestSplit:
    """Tests for train_test_split function."""

    def test_split_by_day(self):
        """Splits by even/odd day of month."""
        from rubin_thermal.data import train_test_split

        # Create sample days
        days = [
            {"date": pd.Timestamp("2024-01-01").date()},  # day 1 (odd) -> test
            {"date": pd.Timestamp("2024-01-02").date()},  # day 2 (even) -> train
            {"date": pd.Timestamp("2024-01-03").date()},  # day 3 (odd) -> test
            {"date": pd.Timestamp("2024-01-04").date()},  # day 4 (even) -> train
        ]

        train, test = train_test_split(days)

        assert len(train) == 2
        assert len(test) == 2

        # Verify correct assignment
        for d in train:
            assert d["date"].day % 2 == 0

        for d in test:
            assert d["date"].day % 2 == 1

    def test_empty_input(self):
        """Handles empty input."""
        from rubin_thermal.data import train_test_split

        train, test = train_test_split([])
        assert len(train) == 0
        assert len(test) == 0
