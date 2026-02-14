"""
Two-zone thermal model for M1M3 mirror with internal gradient tracking.

This module provides a more detailed thermal model that captures the dynamics
between the mirror surface and bulk temperatures. The surface responds faster
to HVAC setpoint changes, while the bulk lags behind, creating transient
internal gradients that can affect image quality.

Physical basis:
- Surface temperature is driven directly by HVAC air temperature (setpoint)
- Bulk temperature responds to surface temperature with a longer time constant
- Internal gradient (T_surface - T_bulk) relaxes over tau_gradient
- Rapid setpoint changes create transient gradients that degrade seeing

The honeycomb structure of M1M3 creates complex thermal patterns, but this
two-zone model captures the dominant first-order behavior.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import CONFIG


@dataclass
class TwoZoneThermalModel:
    """
    Two-zone thermal model: surface and bulk temperature dynamics.

    The model tracks three coupled state variables:
    - T_surface: Temperature at the optical surface (responds to setpoint)
    - T_bulk: Average temperature of the mirror bulk (lags surface)
    - gradient: Internal temperature gradient (T_surface - T_bulk)

    Physics:
    ---------
    Surface dynamics (first-order response to setpoint):
        dT_surface/dt = (T_setpoint - T_surface) / tau_surface

    Bulk dynamics (first-order response to surface):
        dT_bulk/dt = (T_surface - T_bulk) / tau_bulk

    Gradient relaxation:
        The gradient naturally equals T_surface - T_bulk, but we track it
        separately to allow for more complex relaxation dynamics:
        d(gradient)/dt = (T_surface - T_bulk - gradient) / tau_gradient

    Parameters
    ----------
    tau_bulk : float
        Bulk thermal time constant (hours). This is the overall mirror
        thermal mass response. Default: 3.0h from config.
    tau_surface : float
        Surface response time (hours). Surface responds faster than bulk
        because HVAC acts directly on it. Default: 1.0h.
    tau_gradient : float
        Gradient relaxation time (hours). How quickly internal gradients
        dissipate through conduction. Default: 2.5h.
    max_rate : float
        Setpoint rate limit (C/h). Limits how fast setpoint can change.
        Default: 0.7 C/h from config.
    dt : float
        Timestep (hours). Default: 0.25h (15 min) from config.
    cold_bias : float
        Target temperature below ambient (C). Default: 0.3C from config.

    Attributes
    ----------
    T_surface : float
        Current surface temperature (C)
    T_bulk : float
        Current bulk temperature (C)
    gradient : float
        Current internal gradient T_surface - T_bulk (C)
    gradient_integral : float
        Accumulated squared gradient over time (C^2 * h), a damage metric
    T_setpoint_prev : float
        Previous setpoint for rate limiting
    step_count : int
        Number of simulation steps taken
    rate_history : list[float]
        Recent setpoint rates of change (C/h)

    Example
    -------
    >>> model = TwoZoneThermalModel()
    >>> model.initialize(T_initial=10.0)
    >>> for t in np.arange(0, 12, 0.25):
    ...     T_setpoint = 10.0 - 0.5 * t  # Cooling ramp
    ...     T_surf, T_bulk, grad = model.step(T_setpoint)
    ...     print(f"t={t:.1f}h: surf={T_surf:.2f}, bulk={T_bulk:.2f}, grad={grad:.3f}")
    """

    # Model parameters (with defaults from config)
    tau_bulk: float = None
    tau_surface: float = 1.0  # Fast surface response with active cooling (was 1.0)
    tau_gradient: float = 3.0
    max_rate: float = None
    dt: float = None
    cold_bias: float = None
    rate_gradient_coupling: float = 0.3  # Gradient scales with rate: grad += k * rate

    # State variables (initialized in __post_init__)
    T_surface: float = field(default=None, init=False)
    T_bulk: float = field(default=None, init=False)
    gradient: float = field(default=None, init=False)
    gradient_integral: float = field(default=0.0, init=False)
    T_setpoint_prev: float = field(default=None, init=False)
    step_count: int = field(default=0, init=False)
    rate_history: list = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        """Load defaults from config if not specified."""
        physics = CONFIG["physics"]

        if self.tau_bulk is None:
            self.tau_bulk = physics["tau"]
        if self.max_rate is None:
            self.max_rate = physics["max_rate"]
        if self.dt is None:
            self.dt = physics["dt"]
        if self.cold_bias is None:
            self.cold_bias = physics["cold_bias"]

    def initialize(self, T_initial: float, gradient_initial: float = 0.0) -> None:
        """
        Initialize model state for a new simulation run.

        Parameters
        ----------
        T_initial : float
            Initial temperature for both surface and bulk (C)
        gradient_initial : float
            Initial internal gradient (C). Default 0 (isothermal).
            Positive means surface warmer than bulk.
        """
        self.T_surface = T_initial + gradient_initial / 2
        self.T_bulk = T_initial - gradient_initial / 2
        self.gradient = gradient_initial
        self.gradient_integral = 0.0
        self.T_setpoint_prev = T_initial
        self.step_count = 0
        self.rate_history = []
        self._initialized = True

    def reset(self) -> None:
        """
        Reset model to uninitialized state.

        Call initialize() before running a new simulation.
        """
        self.T_surface = None
        self.T_bulk = None
        self.gradient = None
        self.gradient_integral = 0.0
        self.T_setpoint_prev = None
        self.step_count = 0
        self.rate_history = []
        self._initialized = False

    def step(
        self, T_setpoint: float, apply_rate_limit: bool = True
    ) -> tuple[float, float, float]:
        """
        Advance model by one timestep.

        Parameters
        ----------
        T_setpoint : float
            Desired setpoint temperature (C)
        apply_rate_limit : bool
            If True, apply rate limiting to setpoint. Default True.

        Returns
        -------
        tuple : (T_surface, T_bulk, gradient)
            T_surface : float
                New surface temperature (C)
            T_bulk : float
                New bulk temperature (C)
            gradient : float
                New internal gradient (C)

        Raises
        ------
        RuntimeError
            If model has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Apply rate limiting to setpoint
        if apply_rate_limit:
            T_setpoint = self._apply_rate_limit(T_setpoint)

        # Track setpoint rate of change
        setpoint_rate = (T_setpoint - self.T_setpoint_prev) / self.dt
        self.rate_history.append(setpoint_rate)
        if len(self.rate_history) > 24:  # Keep ~6 hours of history at dt=0.25
            self.rate_history.pop(0)

        # Update surface temperature (responds to setpoint)
        dT_surface = (T_setpoint - self.T_surface) / self.tau_surface * self.dt
        T_surface_new = self.T_surface + dT_surface

        # Update bulk temperature (responds to surface)
        dT_bulk = (self.T_surface - self.T_bulk) / self.tau_bulk * self.dt
        T_bulk_new = self.T_bulk + dT_bulk

        # Update gradient with relaxation dynamics
        # Gradient has two components:
        # 1. Equilibrium gradient from surface-bulk difference
        # 2. Rate-dependent gradient from rapid setpoint changes (active cooling effect)
        gradient_equilibrium = T_surface_new - T_bulk_new
        rate_gradient = self.rate_gradient_coupling * abs(
            setpoint_rate
        )  # Rate-dependent term
        gradient_target = gradient_equilibrium + np.sign(setpoint_rate) * rate_gradient
        dGradient = (gradient_target - self.gradient) / self.tau_gradient * self.dt
        gradient_new = self.gradient + dGradient

        # Accumulate gradient damage metric (integral of gradient^2)
        self.gradient_integral += self.gradient**2 * self.dt

        # Update state
        self.T_surface = T_surface_new
        self.T_bulk = T_bulk_new
        self.gradient = gradient_new
        self.T_setpoint_prev = T_setpoint
        self.step_count += 1

        return self.T_surface, self.T_bulk, self.gradient

    def _apply_rate_limit(self, T_setpoint_new: float) -> float:
        """
        Apply rate limiting to setpoint change.

        Parameters
        ----------
        T_setpoint_new : float
            Desired new setpoint

        Returns
        -------
        float
            Rate-limited setpoint
        """
        max_change = self.max_rate * self.dt
        delta = T_setpoint_new - self.T_setpoint_prev
        if abs(delta) > max_change:
            return self.T_setpoint_prev + np.sign(delta) * max_change
        return T_setpoint_new

    def get_gradient(self) -> float:
        """
        Get current internal gradient magnitude.

        Returns
        -------
        float
            Current gradient (T_surface - T_bulk) in C

        Raises
        ------
        RuntimeError
            If model has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self.gradient

    def get_gradient_penalty(self) -> float:
        """
        Get accumulated gradient damage metric.

        This metric quantifies the cumulative effect of internal gradients
        on image quality. It integrates |gradient|^2 over time.

        Units: C^2 * hours

        Interpretation:
        - A constant 0.1C gradient for 10h gives penalty = 0.1 C^2*h
        - A 1.0C gradient spike for 1h gives penalty = 1.0 C^2*h
        - Lower is better; 0 means isothermal mirror throughout

        Returns
        -------
        float
            Accumulated squared gradient integral (C^2 * h)
        """
        return self.gradient_integral

    def get_rms_gradient(self) -> float:
        """
        Get RMS gradient over the simulation so far.

        Returns
        -------
        float
            RMS gradient in C, or 0 if no steps taken
        """
        if self.step_count == 0:
            return 0.0
        time_elapsed = self.step_count * self.dt
        return np.sqrt(self.gradient_integral / time_elapsed)

    def get_average_temperature(self) -> float:
        """
        Get average mirror temperature (mean of surface and bulk).

        Returns
        -------
        float
            Average temperature in C

        Raises
        ------
        RuntimeError
            If model has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return (self.T_surface + self.T_bulk) / 2

    def get_effective_temperature(self, surface_weight: float = 0.7) -> float:
        """
        Get effective mirror temperature for seeing calculation.

        The optical surface dominates image quality, so we weight it higher.

        Parameters
        ----------
        surface_weight : float
            Weight for surface temperature (0-1). Default 0.7.
            Bulk weight is (1 - surface_weight).

        Returns
        -------
        float
            Weighted effective temperature in C

        Raises
        ------
        RuntimeError
            If model has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return surface_weight * self.T_surface + (1 - surface_weight) * self.T_bulk

    def get_state(self) -> dict:
        """
        Get complete model state as a dictionary.

        Returns
        -------
        dict
            Dictionary with all state variables
        """
        return {
            "T_surface": self.T_surface,
            "T_bulk": self.T_bulk,
            "gradient": self.gradient,
            "gradient_integral": self.gradient_integral,
            "T_setpoint_prev": self.T_setpoint_prev,
            "step_count": self.step_count,
            "time_elapsed": self.step_count * self.dt if self._initialized else 0.0,
            "rms_gradient": self.get_rms_gradient() if self._initialized else 0.0,
            "initialized": self._initialized,
        }

    def set_state(self, state: dict) -> None:
        """
        Restore model state from a dictionary.

        Parameters
        ----------
        state : dict
            Dictionary with state variables (from get_state())
        """
        self.T_surface = state["T_surface"]
        self.T_bulk = state["T_bulk"]
        self.gradient = state["gradient"]
        self.gradient_integral = state["gradient_integral"]
        self.T_setpoint_prev = state["T_setpoint_prev"]
        self.step_count = state["step_count"]
        self._initialized = state.get("initialized", True)

    def get_setpoint_rate(self, window: int = 4) -> float:
        """
        Get recent setpoint rate of change.

        Parameters
        ----------
        window : int
            Number of recent steps to average over. Default 4 (1 hour at dt=0.25).

        Returns
        -------
        float
            Average setpoint rate (C/h) over recent window
        """
        if len(self.rate_history) == 0:
            return 0.0
        recent = self.rate_history[-window:]
        return np.mean(recent)


def simulate_with_gradient_model(
    hours: np.ndarray,
    setpoints: np.ndarray,
    T_initial: float,
    model: Optional[TwoZoneThermalModel] = None,
    **model_kwargs,
) -> dict:
    """
    Run a complete simulation with the two-zone gradient model.

    Parameters
    ----------
    hours : np.ndarray
        Time array (hours from some reference, e.g., sunset)
    setpoints : np.ndarray
        Setpoint temperature at each time (C)
    T_initial : float
        Initial mirror temperature (C)
    model : TwoZoneThermalModel, optional
        Model instance to use. If None, creates one with model_kwargs.
    **model_kwargs
        Keyword arguments passed to TwoZoneThermalModel if creating new one.

    Returns
    -------
    dict
        Simulation results with keys:
        - hours: time array
        - T_surface: surface temperature array
        - T_bulk: bulk temperature array
        - T_average: average temperature array
        - gradient: gradient array
        - setpoints: (rate-limited) setpoint array
        - gradient_penalty: final accumulated gradient metric
        - rms_gradient: RMS gradient over simulation
    """
    if model is None:
        model = TwoZoneThermalModel(**model_kwargs)

    model.initialize(T_initial)

    # Interpolate setpoints to model timestep if needed
    dt = model.dt
    t_start, t_end = hours[0], hours[-1]
    sim_times = np.arange(t_start, t_end + dt / 2, dt)
    sim_setpoints = np.interp(sim_times, hours, setpoints)

    # Allocate output arrays
    n = len(sim_times)
    T_surface_out = np.zeros(n)
    T_bulk_out = np.zeros(n)
    gradient_out = np.zeros(n)
    setpoint_out = np.zeros(n)

    # Initial conditions
    T_surface_out[0] = model.T_surface
    T_bulk_out[0] = model.T_bulk
    gradient_out[0] = model.gradient
    setpoint_out[0] = T_initial

    # Run simulation
    for i in range(1, n):
        T_surf, T_bulk, grad = model.step(sim_setpoints[i])
        T_surface_out[i] = T_surf
        T_bulk_out[i] = T_bulk
        gradient_out[i] = grad
        setpoint_out[i] = model.T_setpoint_prev

    return {
        "hours": sim_times,
        "T_surface": T_surface_out,
        "T_bulk": T_bulk_out,
        "T_average": (T_surface_out + T_bulk_out) / 2,
        "gradient": gradient_out,
        "setpoints": setpoint_out,
        "gradient_penalty": model.get_gradient_penalty(),
        "rms_gradient": model.get_rms_gradient(),
    }


def compare_gradient_impact(
    hours: np.ndarray,
    setpoints_fast: np.ndarray,
    setpoints_slow: np.ndarray,
    T_initial: float,
    **model_kwargs,
) -> dict:
    """
    Compare gradient impact of two different setpoint strategies.

    Useful for evaluating whether a more aggressive control strategy
    creates unacceptable internal gradients.

    Parameters
    ----------
    hours : np.ndarray
        Time array
    setpoints_fast : np.ndarray
        Setpoints for faster/more aggressive strategy
    setpoints_slow : np.ndarray
        Setpoints for slower/conservative strategy
    T_initial : float
        Initial mirror temperature
    **model_kwargs
        Keyword arguments passed to TwoZoneThermalModel.

    Returns
    -------
    dict
        Comparison results with keys:
        - fast: simulation results for fast strategy
        - slow: simulation results for slow strategy
        - gradient_ratio: ratio of gradient penalties (fast/slow)
        - rms_ratio: ratio of RMS gradients (fast/slow)
    """
    results_fast = simulate_with_gradient_model(
        hours, setpoints_fast, T_initial, **model_kwargs
    )
    results_slow = simulate_with_gradient_model(
        hours, setpoints_slow, T_initial, **model_kwargs
    )

    # Compute comparison metrics
    penalty_fast = results_fast["gradient_penalty"]
    penalty_slow = results_slow["gradient_penalty"]
    gradient_ratio = penalty_fast / penalty_slow if penalty_slow > 0 else np.inf

    rms_fast = results_fast["rms_gradient"]
    rms_slow = results_slow["rms_gradient"]
    rms_ratio = rms_fast / rms_slow if rms_slow > 0 else np.inf

    return {
        "fast": results_fast,
        "slow": results_slow,
        "gradient_ratio": gradient_ratio,
        "rms_ratio": rms_ratio,
    }
