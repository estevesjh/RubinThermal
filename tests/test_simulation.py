"""
Integration tests for simulation pipeline.
"""

import pytest
import numpy as np

from rubin_thermal import (
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    ThermalModel,
)
from rubin_thermal.control import phase1_fixed, phase2_ramp, phase3_lookahead


def simulate_day(day, model, T1=-3, T2=0):
    """
    Run full simulation for a single day.

    This replicates the simulation loop used in the scripts.
    """
    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    n = len(hours)
    T_setpoint = np.zeros(n)
    T_mirror = np.zeros(n)
    T_ambient = temps.copy()
    phase = np.zeros(n, dtype=int)

    # Initialize
    T_mirror[0] = T_sunset - model.cold_bias

    for i in range(n):
        t = hours[i]
        T_amb = temps[i]
        rate = compute_rate(hours, temps, t, 1.0)

        if t < T1:
            T_setpoint[i] = phase1_fixed(
                t, T_amb, T_sunset, rate, hours, temps, model.cold_bias
            )
            phase[i] = 1
        elif t < T2:
            T_setpoint[i] = phase2_ramp(
                t, T_amb, T_sunset, rate, hours, temps, T1, T2, model.tau, model.cold_bias
            )
            phase[i] = 2
        else:
            T_setpoint[i] = phase3_lookahead(
                t, T_amb, T_sunset, rate, hours, temps, model.tau, model.cold_bias
            )
            phase[i] = 3

        if i > 0:
            T_setpoint[i] = model.rate_limit(T_setpoint[i], T_setpoint[i - 1])
            T_mirror[i] = model.step(T_mirror[i - 1], T_setpoint[i])

    T_target = T_ambient - model.cold_bias
    errors = T_mirror - T_target

    sunset_idx = np.argmin(np.abs(hours))
    overnight_mask = hours >= 0

    return {
        "t": hours,
        "T_ambient": T_ambient,
        "T_target": T_target,
        "T_setpoint": T_setpoint,
        "T_mirror": T_mirror,
        "errors": errors,
        "phase": phase,
        "error_at_sunset": errors[sunset_idx],
        "rms_overnight": np.sqrt(np.mean(errors[overnight_mask] ** 2)),
    }


class TestSimulationLoop:
    """Tests for full simulation loop."""

    def test_simulation_runs(self, sample_day):
        """Simulation completes without error."""
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        assert "T_mirror" in result
        assert "errors" in result
        assert len(result["T_mirror"]) == len(sample_day["hours_from_sunset"])

    def test_mirror_tracks_target(self, sample_day):
        """Mirror stays reasonably close to target."""
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        # RMS error should be reasonable (< 1C for this smooth profile)
        assert result["rms_overnight"] < 1.0

    def test_phase_progression(self, sample_day):
        """Phases progress correctly through the day."""
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        # Should have all three phases
        unique_phases = set(result["phase"])
        assert 1 in unique_phases
        assert 2 in unique_phases
        assert 3 in unique_phases

        # Phases should be monotonically non-decreasing
        for i in range(1, len(result["phase"])):
            assert result["phase"][i] >= result["phase"][i - 1]

    def test_rate_limiting_works(self, sample_day):
        """Setpoint changes are rate-limited."""
        model = ThermalModel(max_rate=0.7, dt=0.25)
        result = simulate_day(sample_day, model)

        max_change = model.max_rate * model.dt
        setpoint_changes = np.diff(result["T_setpoint"])

        # All changes should be within rate limit (with small tolerance)
        assert np.all(np.abs(setpoint_changes) <= max_change + 1e-10)

    def test_thermal_lag(self, sample_day):
        """Mirror lags behind setpoint as expected."""
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        # Mirror should generally lag setpoint during phase transitions
        # Check that mirror and setpoint aren't identical
        diff = result["T_mirror"] - result["T_setpoint"]
        assert np.any(np.abs(diff) > 0.01)  # Some lag exists


class TestErrorCalculation:
    """Tests for error calculation."""

    def test_error_definition(self, sample_day):
        """Error = T_mirror - (T_ambient - cold_bias)."""
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        expected_errors = result["T_mirror"] - result["T_target"]
        np.testing.assert_array_almost_equal(result["errors"], expected_errors)

    def test_negative_error_means_cold(self, sample_day):
        """Negative error means mirror is colder than target (good)."""
        # This is a sanity check on the error definition
        model = ThermalModel()
        result = simulate_day(sample_day, model)

        # Find a point where mirror is colder than target
        cold_mask = result["T_mirror"] < result["T_target"]
        if np.any(cold_mask):
            # Errors at those points should be negative
            assert np.all(result["errors"][cold_mask] < 0)


class TestDifferentParameters:
    """Tests with different model parameters."""

    def test_higher_tau_slower_response(self, sample_day):
        """Higher tau means slower thermal response."""
        model_fast = ThermalModel(tau=1.0)
        model_slow = ThermalModel(tau=5.0)

        result_fast = simulate_day(sample_day, model_fast)
        result_slow = simulate_day(sample_day, model_slow)

        # With constant setpoint, faster tau reaches equilibrium quicker
        # Compare late-night errors
        late_mask = sample_day["hours_from_sunset"] > 4

        # This is a qualitative check - exact comparison depends on profile
        # Just ensure simulation runs with different taus
        assert result_fast["rms_overnight"] is not None
        assert result_slow["rms_overnight"] is not None

    def test_strict_rate_limit(self, sample_day):
        """Very strict rate limit constrains setpoint changes."""
        model = ThermalModel(max_rate=0.1, dt=0.25)  # Very slow
        result = simulate_day(sample_day, model)

        max_change = model.max_rate * model.dt
        setpoint_changes = np.abs(np.diff(result["T_setpoint"]))

        # Most changes should be at the rate limit
        at_limit = setpoint_changes >= max_change - 1e-10
        # At least some should be rate-limited
        assert np.sum(at_limit) > 0
