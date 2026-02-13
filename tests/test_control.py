"""
Unit tests for rubin_thermal.control module.
"""

import pytest
import numpy as np

from rubin_thermal.control import (
    phase1_fixed,
    phase2_ramp,
    phase3_lookahead,
    three_phase_setpoint,
    PHASE2_OPTIONS,
)


class TestPhase1Fixed:
    """Tests for phase1_fixed function."""

    def test_basic_calculation(self):
        """Returns T_sunset - cold_bias."""
        hours = np.array([-6, -4, -2, 0, 2])
        temps = np.array([18, 17, 16, 15, 14])

        result = phase1_fixed(
            t=-5.0,
            T_amb=17.5,
            T_sunset=15.0,
            rate=-0.5,
            hours=hours,
            temps=temps,
            cold_bias=0.3,
        )

        assert abs(result - 14.7) < 1e-10

    def test_ignores_ambient_and_rate(self):
        """Result independent of T_amb and rate."""
        hours = np.array([-6, -4, -2, 0, 2])
        temps = np.array([18, 17, 16, 15, 14])

        # Different ambient and rate values
        r1 = phase1_fixed(-5, 20.0, 15.0, 0.0, hours, temps, 0.3)
        r2 = phase1_fixed(-5, 10.0, 15.0, -2.0, hours, temps, 0.3)

        assert r1 == r2


class TestPhase2Ramp:
    """Tests for phase2_ramp function."""

    def test_at_t1_equals_fixed(self):
        """At T1, should equal fixed setpoint."""
        hours = np.array([-6, -4, -2, 0, 2])
        temps = np.array([18, 17, 16, 15, 14])

        result = phase2_ramp(
            t=-3.0,  # = T1
            T_amb=16.5,
            T_sunset=15.0,
            rate=-0.5,
            hours=hours,
            temps=temps,
            T1=-3.0,
            T2=0.0,
            tau=3.0,
            cold_bias=0.3,
        )

        # At T1, alpha=0, so should be T_sunset - cold_bias
        expected = 15.0 - 0.3
        assert abs(result - expected) < 1e-10

    def test_at_t2_equals_tracking(self):
        """At T2, should equal tracking setpoint."""
        hours = np.array([-6, -4, -2, 0, 2])
        temps = np.array([18, 17, 16, 15, 14])

        T_amb = 15.0
        rate = -0.5
        tau = 3.0
        cold_bias = 0.3

        result = phase2_ramp(
            t=0.0,  # = T2
            T_amb=T_amb,
            T_sunset=15.0,
            rate=rate,
            hours=hours,
            temps=temps,
            T1=-3.0,
            T2=0.0,
            tau=tau,
            cold_bias=cold_bias,
        )

        # At T2, alpha=1, so should be T_amb - cold_bias + 0.5*tau*rate
        expected = T_amb - cold_bias + 0.5 * tau * rate
        assert abs(result - expected) < 1e-10


class TestPhase3Lookahead:
    """Tests for phase3_lookahead function."""

    def test_uses_weighted_average(self):
        """Uses weighted average of future temperatures."""
        # Create flat temperature profile
        hours = np.arange(0, 10, 0.25)
        temps = np.full_like(hours, 15.0)

        result = phase3_lookahead(
            t=2.0,
            T_amb=15.0,
            T_sunset=16.0,
            rate=0.0,
            hours=hours,
            temps=temps,
            tau=3.0,
            cold_bias=0.3,
        )

        # With flat temps and zero rate:
        # weighted_temp = 15.0, result = 15.0 + 0 - 0.3 = 14.7
        expected = 15.0 - 0.3
        assert abs(result - expected) < 1e-10


class TestThreePhaseSetpoint:
    """Tests for three_phase_setpoint function."""

    def test_returns_phase(self):
        """Returns both setpoint and phase number."""
        hours = np.arange(-8, 10, 0.25)
        temps = np.full_like(hours, 15.0)

        setpoint, phase = three_phase_setpoint(
            t=-5.0,
            T_amb=15.0,
            T_sunset=15.0,
            rate=0.0,
            hours=hours,
            temps=temps,
        )

        assert isinstance(setpoint, float)
        assert phase in [1, 2, 3]

    def test_phase_transitions(self):
        """Correct phase at different times."""
        hours = np.arange(-8, 10, 0.25)
        temps = np.full_like(hours, 15.0)

        # Phase 1 (before T1=-3)
        _, phase = three_phase_setpoint(-5.0, 15.0, 15.0, 0.0, hours, temps)
        assert phase == 1

        # Phase 2 (between T1=-3 and T2=0)
        _, phase = three_phase_setpoint(-1.5, 15.0, 15.0, 0.0, hours, temps)
        assert phase == 2

        # Phase 3 (after T2=0)
        _, phase = three_phase_setpoint(2.0, 15.0, 15.0, 0.0, hours, temps)
        assert phase == 3


class TestPhase2Options:
    """Tests for PHASE2_OPTIONS dictionary."""

    def test_all_options_callable(self):
        """All phase 2 options are callable functions."""
        for name, func in PHASE2_OPTIONS.items():
            assert callable(func)

    def test_expected_options_present(self):
        """Expected algorithm names are present."""
        assert "Ramp fixed->track" in PHASE2_OPTIONS
        assert "Track + 0.5*tau*rate" in PHASE2_OPTIONS
