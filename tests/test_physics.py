"""
Unit tests for rubin_thermal.physics module.
"""

import pytest
import numpy as np

from rubin_thermal.physics import (
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    thermal_step,
    apply_rate_limit,
    ThermalModel,
)


class TestInterpolateTemp:
    """Tests for interpolate_temp function."""

    def test_exact_value(self, sample_time_series):
        """Interpolation at exact data point."""
        hours, temps = sample_time_series
        assert interpolate_temp(hours, temps, 0.0) == 15.0
        assert interpolate_temp(hours, temps, 2.0) == 13.0

    def test_midpoint(self, sample_time_series):
        """Interpolation at midpoint between data points."""
        hours, temps = sample_time_series
        # Between 0 (15.0) and 1 (14.0), midpoint should be 14.5
        result = interpolate_temp(hours, temps, 0.5)
        assert abs(result - 14.5) < 1e-10

    def test_before_start(self, sample_time_series):
        """Clamps to first value before data start."""
        hours, temps = sample_time_series
        result = interpolate_temp(hours, temps, -1.0)
        assert result == temps[0]

    def test_after_end(self, sample_time_series):
        """Clamps to last value after data end."""
        hours, temps = sample_time_series
        result = interpolate_temp(hours, temps, 10.0)
        assert result == temps[-1]


class TestComputeRate:
    """Tests for compute_rate function."""

    def test_constant_temperature(self):
        """Rate is zero for constant temperature."""
        hours = np.array([0, 1, 2, 3, 4])
        temps = np.array([15, 15, 15, 15, 15])
        rate = compute_rate(hours, temps, 2.0, window=1.0)
        assert abs(rate) < 1e-10

    def test_linear_cooling(self):
        """Rate matches slope for linear cooling."""
        hours = np.array([0, 1, 2, 3, 4])
        temps = np.array([20, 19, 18, 17, 16])  # -1 C/hour
        rate = compute_rate(hours, temps, 2.0, window=1.0)
        assert abs(rate - (-1.0)) < 1e-10

    def test_edge_clamping(self):
        """Handles edge cases gracefully."""
        hours = np.array([0, 1, 2])
        temps = np.array([15, 14, 13])
        # At edge, window is truncated
        rate = compute_rate(hours, temps, 0.0, window=2.0)
        # Should still compute some rate
        assert isinstance(rate, float)


class TestMakeLinearKernel:
    """Tests for make_linear_kernel function."""

    def test_weights_sum_to_one(self):
        """Kernel weights must sum to 1."""
        weights, times = make_linear_kernel(13, 3.0)
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_correct_length(self):
        """Returns correct number of points."""
        weights, times = make_linear_kernel(13, 3.0)
        assert len(weights) == 13
        assert len(times) == 13

    def test_times_range(self):
        """Times span from 0 to T_max."""
        weights, times = make_linear_kernel(13, 3.0)
        assert times[0] == 0.0
        assert times[-1] == 3.0

    def test_weights_decrease(self):
        """Weights decrease linearly (triangular kernel)."""
        weights, times = make_linear_kernel(5, 4.0)
        # First weight should be largest
        assert weights[0] > weights[-1]
        # Weights should decrease monotonically
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_default_values(self):
        """Uses config defaults when not specified."""
        weights, times = make_linear_kernel()
        # Should return valid kernel
        assert len(weights) > 0
        assert abs(sum(weights) - 1.0) < 1e-10


class TestThermalStep:
    """Tests for thermal_step function."""

    def test_no_change_at_equilibrium(self):
        """No change when mirror equals setpoint."""
        T_new = thermal_step(15.0, 15.0, tau=3.0, dt=0.25)
        assert abs(T_new - 15.0) < 1e-10

    def test_approaches_setpoint(self):
        """Mirror temperature moves toward setpoint."""
        T_mirror = 10.0
        T_setpoint = 15.0
        T_new = thermal_step(T_mirror, T_setpoint, tau=3.0, dt=0.25)
        # Should be between current and setpoint
        assert T_new > T_mirror
        assert T_new < T_setpoint

    def test_known_value(self):
        """Verify exact calculation: T_new = T_old + (T_sp - T_old) / tau * dt."""
        # With tau=3.0, dt=0.25, T_mirror=10, T_setpoint=13:
        # Expected: 10 + (13-10)/3.0 * 0.25 = 10.25
        T_new = thermal_step(10.0, 13.0, tau=3.0, dt=0.25)
        assert abs(T_new - 10.25) < 1e-10


class TestApplyRateLimit:
    """Tests for apply_rate_limit function."""

    def test_within_limit(self):
        """No change when within rate limit."""
        # max_rate=0.7, dt=0.25 -> max_change=0.175
        result = apply_rate_limit(15.1, 15.0, max_rate=0.7, dt=0.25)
        assert result == 15.1

    def test_positive_limit_exceeded(self):
        """Clamps positive change at limit."""
        # max_rate=0.7, dt=0.25 -> max_change=0.175
        result = apply_rate_limit(16.0, 15.0, max_rate=0.7, dt=0.25)
        assert abs(result - 15.175) < 1e-10

    def test_negative_limit_exceeded(self):
        """Clamps negative change at limit."""
        result = apply_rate_limit(14.0, 15.0, max_rate=0.7, dt=0.25)
        assert abs(result - 14.825) < 1e-10


class TestThermalModel:
    """Tests for ThermalModel class."""

    def test_default_initialization(self):
        """Loads defaults from config."""
        model = ThermalModel()
        assert model.tau is not None
        assert model.max_rate is not None
        assert model.cold_bias is not None
        assert model.dt is not None

    def test_custom_initialization(self):
        """Accepts custom parameters."""
        model = ThermalModel(tau=2.0, max_rate=1.0, cold_bias=0.5, dt=0.5)
        assert model.tau == 2.0
        assert model.max_rate == 1.0
        assert model.cold_bias == 0.5
        assert model.dt == 0.5

    def test_step_method(self):
        """Step method uses instance parameters."""
        model = ThermalModel(tau=3.0, dt=0.25)
        T_new = model.step(10.0, 13.0)
        # Same calculation as test_known_value
        assert abs(T_new - 10.25) < 1e-10

    def test_rate_limit_method(self):
        """Rate limit method uses instance parameters."""
        model = ThermalModel(max_rate=0.7, dt=0.25)
        result = model.rate_limit(16.0, 15.0)
        assert abs(result - 15.175) < 1e-10
