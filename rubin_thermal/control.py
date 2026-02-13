"""
Control algorithms for three-phase mirror thermal management.

Phase 1: DAYTIME - Fixed setpoint based on predicted sunset temperature
Phase 2: PRE-SUNSET TRANSITION - Transition from fixed to tracking
Phase 3: OVERNIGHT - Lookahead tracking with rate compensation
"""

import numpy as np

from .config import CONFIG
from .physics import interpolate_temp, compute_rate, make_linear_kernel


def phase1_fixed(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 1: Fixed setpoint at predicted sunset temperature minus cold bias.

    Parameters
    ----------
    t : float
        Time relative to sunset (hours)
    T_amb : float
        Current ambient temperature
    T_sunset : float
        Predicted temperature at sunset
    rate : float
        Current temperature rate of change
    hours : np.ndarray
        Time array for interpolation
    temps : np.ndarray
        Temperature array for interpolation
    cold_bias : float, optional
        Cold bias offset. Defaults to config value.

    Returns
    -------
    float
        Setpoint temperature
    """
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]
    return T_sunset - cold_bias


def phase2_ramp(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    T1: float,
    T2: float,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 2: Linear ramp from fixed setpoint to tracking.

    Parameters
    ----------
    t : float
        Time relative to sunset (hours)
    T_amb : float
        Current ambient temperature
    T_sunset : float
        Predicted temperature at sunset
    rate : float
        Current temperature rate of change
    hours : np.ndarray
        Time array for interpolation
    temps : np.ndarray
        Temperature array for interpolation
    T1 : float
        Start of phase 2 (hours before sunset, negative)
    T2 : float
        End of phase 2 (hours before sunset)
    tau : float, optional
        Thermal time constant. Defaults to config value.
    cold_bias : float, optional
        Cold bias offset. Defaults to config value.

    Returns
    -------
    float
        Setpoint temperature
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    alpha = (t - T1) / (T2 - T1)
    fixed = T_sunset - cold_bias
    tracking = T_amb - cold_bias + 0.5 * tau * rate
    return (1 - alpha) * fixed + alpha * tracking


def phase2_track_rate(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    T1: float,
    T2: float,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 2: Track ambient with 0.5*tau rate compensation.

    Parameters
    ----------
    (same as phase2_ramp)

    Returns
    -------
    float
        Setpoint temperature
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    return T_amb - cold_bias + 0.5 * tau * rate


def phase2_aggressive_track(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    T1: float,
    T2: float,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 2: Aggressive tracking with full tau rate compensation.

    Parameters
    ----------
    (same as phase2_ramp)

    Returns
    -------
    float
        Setpoint temperature
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    return T_amb - cold_bias + tau * rate


def phase2_lookahead_ramp(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    T1: float,
    T2: float,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 2: Gradually introduce lookahead from simple to weighted.

    Parameters
    ----------
    (same as phase2_ramp)

    Returns
    -------
    float
        Setpoint temperature
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    alpha = (t - T1) / (T2 - T1)
    simple = T_amb - cold_bias + 0.5 * tau * rate

    weights, times = make_linear_kernel()
    weighted_temp = sum(
        w * interpolate_temp(hours, temps, t + dt) for w, dt in zip(weights, times)
    )
    lookahead = weighted_temp + 0.5 * tau * rate - cold_bias

    return (1 - alpha) * simple + alpha * lookahead


def phase3_lookahead(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> float:
    """
    Phase 3: Linear weighted lookahead with rate compensation.

    Uses a triangular kernel over the next 3 hours (configurable)
    plus rate compensation to anticipate temperature changes.

    Parameters
    ----------
    t : float
        Time relative to sunset (hours)
    T_amb : float
        Current ambient temperature
    T_sunset : float
        Predicted temperature at sunset
    rate : float
        Current temperature rate of change
    hours : np.ndarray
        Time array for interpolation
    temps : np.ndarray
        Temperature array for interpolation
    tau : float, optional
        Thermal time constant. Defaults to config value.
    cold_bias : float, optional
        Cold bias offset. Defaults to config value.

    Returns
    -------
    float
        Setpoint temperature
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    weights, times = make_linear_kernel()
    weighted_temp = sum(
        w * interpolate_temp(hours, temps, t + dt) for w, dt in zip(weights, times)
    )
    return weighted_temp + 0.5 * tau * rate - cold_bias


def three_phase_setpoint(
    t: float,
    T_amb: float,
    T_sunset: float,
    rate: float,
    hours: np.ndarray,
    temps: np.ndarray,
    T1: float | None = None,
    T2: float | None = None,
    tau: float | None = None,
    cold_bias: float | None = None,
) -> tuple[float, int]:
    """
    Compute setpoint using optimal three-phase control.

    Parameters
    ----------
    t : float
        Time relative to sunset (hours)
    T_amb : float
        Current ambient temperature
    T_sunset : float
        Predicted temperature at sunset
    rate : float
        Current temperature rate of change (C/hour)
    hours : np.ndarray
        Full time array for the day
    temps : np.ndarray
        Full temperature array for the day
    T1 : float, optional
        Phase 2 start (hours before sunset). Defaults to config.
    T2 : float, optional
        Phase 3 start (hours before sunset). Defaults to config.
    tau : float, optional
        Thermal time constant. Defaults to config.
    cold_bias : float, optional
        Cold bias offset. Defaults to config.

    Returns
    -------
    tuple (setpoint, phase)
        setpoint: temperature setpoint
        phase: 1, 2, or 3
    """
    if T1 is None:
        T1 = CONFIG["control"]["t1"]
    if T2 is None:
        T2 = CONFIG["control"]["t2"]
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if cold_bias is None:
        cold_bias = CONFIG["physics"]["cold_bias"]

    if t < T1:
        # Phase 1: Daytime fixed
        setpoint = phase1_fixed(t, T_amb, T_sunset, rate, hours, temps, cold_bias)
        phase = 1
    elif t < T2:
        # Phase 2: Pre-sunset transition (using ramp, the best performer)
        setpoint = phase2_ramp(
            t, T_amb, T_sunset, rate, hours, temps, T1, T2, tau, cold_bias
        )
        phase = 2
    else:
        # Phase 3: Overnight lookahead
        setpoint = phase3_lookahead(
            t, T_amb, T_sunset, rate, hours, temps, tau, cold_bias
        )
        phase = 3

    return setpoint, phase


# Dictionary of available phase 2 algorithms for optimization
PHASE2_OPTIONS = {
    "Ramp fixed->track": phase2_ramp,
    "Track + 0.5*tau*rate": phase2_track_rate,
    "Track + tau*rate": phase2_aggressive_track,
    "Lookahead ramp": phase2_lookahead_ramp,
}
