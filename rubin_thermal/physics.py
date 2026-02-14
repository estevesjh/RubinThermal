"""
Thermal physics model for RubinThermal.

Consolidates thermal dynamics, interpolation, and rate computation
that was previously duplicated across scripts.
"""

from typing import TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

from .config import CONFIG

if TYPE_CHECKING:
    from .forecast import TemperatureProvider


def interpolate_temp(hours: np.ndarray, temps: np.ndarray, t: float) -> float:
    """
    Interpolate temperature at time t from time series.

    Handles edge cases by clamping to boundary values.

    Parameters
    ----------
    hours : np.ndarray
        Time values (hours from sunset)
    temps : np.ndarray
        Temperature values
    t : float
        Time to interpolate at

    Returns
    -------
    float
        Interpolated temperature
    """
    if t <= hours[0]:
        return temps[0]
    if t >= hours[-1]:
        return temps[-1]
    return np.interp(t, hours, temps)


def compute_rate(
    hours: np.ndarray,
    temps: np.ndarray,
    t: float,
    window: float = 1.0,
    forecast_provider: "TemperatureProvider | None" = None,
    t_origin: float | None = None,
) -> float:
    """
    Compute temperature rate of change at time t.

    Uses centered difference over a window.

    Parameters
    ----------
    hours : np.ndarray
        Time values
    temps : np.ndarray
        Temperature values
    t : float
        Time at which to compute rate
    window : float
        Window size in hours for rate computation
    forecast_provider : TemperatureProvider, optional
        Provider for future temperature values. If None, uses actual data.
    t_origin : float, optional
        Time origin for forecasts. Required if forecast_provider is set.

    Returns
    -------
    float
        Rate of change in C/hour
    """
    t_before = max(t - window / 2, hours[0])
    t_after = min(t + window / 2, hours[-1])
    if t_after <= t_before:
        return 0.0

    if forecast_provider is not None and t_origin is not None:
        T_before = forecast_provider.get_temperature(t_before, t_origin, hours, temps)
        T_after = forecast_provider.get_temperature(t_after, t_origin, hours, temps)
    else:
        T_before = interpolate_temp(hours, temps, t_before)
        T_after = interpolate_temp(hours, temps, t_after)
    return (T_after - T_before) / (t_after - t_before)


def make_linear_kernel(
    n_points: int | None = None, T_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create linear weighting kernel for lookahead.

    Weights decrease linearly from 1 at t=0 to 0 at t=T_max.

    Parameters
    ----------
    n_points : int, optional
        Number of points in kernel. Defaults to config value.
    T_max : float, optional
        Maximum lookahead time in hours. Defaults to config value.

    Returns
    -------
    tuple (weights, times)
        weights: normalized weights summing to 1
        times: time offsets in hours
    """
    if n_points is None:
        n_points = CONFIG["control"]["lookahead_points"]
    if T_max is None:
        T_max = CONFIG["control"]["lookahead_hours"]

    times = np.linspace(0, T_max, n_points)
    weights = (T_max - times) / T_max
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return weights, times


def thermal_step(
    T_mirror: float,
    T_setpoint: float,
    tau: float | None = None,
    dt: float | None = None,
) -> float:
    """
    Compute single step of thermal dynamics.

    First-order response: dT_mirror/dt = (T_setpoint - T_mirror) / tau

    Parameters
    ----------
    T_mirror : float
        Current mirror temperature
    T_setpoint : float
        Current setpoint temperature
    tau : float, optional
        Time constant in hours. Defaults to config value.
    dt : float, optional
        Timestep in hours. Defaults to config value.

    Returns
    -------
    float
        New mirror temperature after one timestep
    """
    if tau is None:
        tau = CONFIG["physics"]["tau"]
    if dt is None:
        dt = CONFIG["physics"]["dt"]

    dT = (T_setpoint - T_mirror) / tau * dt
    return T_mirror + dT


def apply_rate_limit(
    T_setpoint_new: float,
    T_setpoint_prev: float,
    max_rate: float | None = None,
    dt: float | None = None,
) -> float:
    """
    Apply rate limiting to setpoint change.

    Parameters
    ----------
    T_setpoint_new : float
        Desired new setpoint
    T_setpoint_prev : float
        Previous setpoint
    max_rate : float, optional
        Maximum rate of change in C/hour. Defaults to config value.
    dt : float, optional
        Timestep in hours. Defaults to config value.

    Returns
    -------
    float
        Rate-limited setpoint
    """
    if max_rate is None:
        max_rate = CONFIG["physics"]["max_rate"]
    if dt is None:
        dt = CONFIG["physics"]["dt"]

    max_change = max_rate * dt
    delta = T_setpoint_new - T_setpoint_prev
    if abs(delta) > max_change:
        return T_setpoint_prev + np.sign(delta) * max_change
    return T_setpoint_new


@dataclass
class ThermalModel:
    """
    Encapsulates thermal model parameters.

    Attributes
    ----------
    tau : float
        Mirror thermal time constant (hours)
    max_rate : float
        Maximum setpoint change rate (C/hour)
    cold_bias : float
        Target temperature below ambient (C)
    dt : float
        Simulation timestep (hours)
    """

    tau: float = None
    max_rate: float = None
    cold_bias: float = None
    dt: float = None

    def __post_init__(self):
        """Load defaults from config if not specified."""
        physics = CONFIG["physics"]
        if self.tau is None:
            self.tau = physics["tau"]
        if self.max_rate is None:
            self.max_rate = physics["max_rate"]
        if self.cold_bias is None:
            self.cold_bias = physics["cold_bias"]
        if self.dt is None:
            self.dt = physics["dt"]

    def step(self, T_mirror: float, T_setpoint: float) -> float:
        """Compute thermal response for one timestep."""
        return thermal_step(T_mirror, T_setpoint, self.tau, self.dt)

    def rate_limit(self, T_new: float, T_prev: float) -> float:
        """Apply rate limiting to setpoint change."""
        return apply_rate_limit(T_new, T_prev, self.max_rate, self.dt)
