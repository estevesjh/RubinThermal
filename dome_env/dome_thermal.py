"""
Dome thermal environment analysis module.

Characterizes how the dome interior thermalizes after opening using
ODE forward-modeling against real temperature data.

Models:
    0. Simple exponential decay (baseline)
    1. Basic ODE with moving ambient target
    2. Derivative-coupled ODE (handles volatile weather)
    3. Deep-mass coupled ODE (physical: air + structural thermal mass)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

DATA_PATH = "/sdf/data/rubin/user/esteves/thermal_analysis/thermal_data_all_observing_nights.csv"

NEEDED_COLS = [
    "inside_m2_temp_mean",
    "outside_temp_mean",
    "dome_open",
    "sunAltitude",
    "wind_speed_mean",
    "night_date",
]

# ESS 113 is loaded separately and renamed to ess113_temp_mean
ESS113_COL = "inside_m1m3_temp_mean"

SONIC_PATH = "/sdf/data/rubin/user/esteves/thermal_analysis/sonic_data_observing_nights.csv"


def load_thermal_data(csv_path=DATA_PATH, sonic_path=SONIC_PATH):
    """Load thermal dataset with ESS 113 renamed to ess113_temp_mean."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True, low_memory=False)

    # Rename ESS 113 column
    if ESS113_COL in df.columns:
        df = df.rename(columns={ESS113_COL: "ess113_temp_mean"})

    available = [c for c in NEEDED_COLS + ["ess113_temp_mean"] if c in df.columns]
    df = df[available]

    return df


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def smooth_temperatures(series, window_min=61, polyorder=3):
    """
    Apply Savitzky-Golay filter to a temperature series.

    Unlike a moving average, Savgol preserves the steep initial drop
    when the dome opens while removing high-frequency turbulence noise.

    Parameters
    ----------
    series : array-like
        Temperature values at 1-min cadence.
    window_min : int
        Filter window in minutes (must be odd). Default 61 = 1 hour.
    polyorder : int
        Polynomial order for the filter.

    Returns
    -------
    np.ndarray
        Smoothed temperature series.
    """
    arr = np.asarray(series, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < window_min:
        return arr

    # Interpolate gaps for filtering, then restore NaNs
    interp = pd.Series(arr).interpolate(limit_direction="both").values
    smoothed = savgol_filter(interp, window_min, polyorder)
    smoothed[~mask] = np.nan
    return smoothed


def get_dome_open_time(night_df):
    """
    Find the dome-open time for a night.

    Returns the first timestamp where dome_open == True, as-is.

    Returns
    -------
    pd.Timestamp or None
    """
    if "dome_open" not in night_df.columns:
        return None

    open_mask = night_df["dome_open"] == True
    if not open_mask.any():
        return None

    return night_df.index[open_mask][0]


def prepare_night(night_df, eval_hours=9.0, pre_open_min=60,
                  min_dome_hours=3.0):
    """
    Full preprocessing pipeline for one night.

    1. Require >= min_dome_hours of dome-open time
    2. Smooth inside and outside temps independently
    3. Find dome-open time (with lookback correction)
    4. Extract window: 1h before dome-open to +9h after
    5. Compute time array in hours from dome-open (t=0)

    Parameters
    ----------
    night_df : DataFrame
        Single night data.
    eval_hours : float
        Hours after dome-open to include (default 9).
    pre_open_min : int
        Minutes before dome-open to include (default 30).
    min_dome_hours : float
        Minimum hours of dome-open required (default 3).

    Returns
    -------
    dict or None
        Keys: t (hours), T_in, T_out, T_in_raw, T_out_raw,
              t_open, wind_speed, night_date, dome_hours
    """
    required = ["inside_m2_temp_mean", "outside_temp_mean"]
    for col in required:
        if col not in night_df.columns:
            return None

    # Check dome-open duration (in hours)
    if "dome_open" in night_df.columns:
        dome_open_minutes = (night_df["dome_open"] == True).sum()  # 1-min resolution
        dome_hours = dome_open_minutes / 60.0
        if dome_hours < min_dome_hours:
            return None
    else:
        dome_hours = np.nan

    # Drop rows missing key temps
    valid = night_df[required].dropna()
    if len(valid) < 120:  # need at least 2 hours of data
        return None

    # Smooth temperatures
    # Inside: 15-min Savgol (light smoothing, preserves structure response)
    # Outside: 1-hour Savgol (heavier smoothing, needed for clean dT_out/dt)
    T_in_raw = night_df["inside_m2_temp_mean"].values
    T_out_raw = night_df["outside_temp_mean"].values
    T_in_smooth = smooth_temperatures(T_in_raw, window_min=15)
    T_out_smooth = smooth_temperatures(T_out_raw, window_min=61)

    # Find dome-open time
    t_open = get_dome_open_time(night_df)
    if t_open is None:
        return None

    # Extract window: 30 min before dome-open to +eval_hours after
    t_start = t_open - pd.Timedelta(minutes=pre_open_min)
    t_end = t_open + pd.Timedelta(hours=eval_hours)
    # Clamp to data range
    t_start = max(t_start, night_df.index[0])
    mask = (night_df.index >= t_start) & (night_df.index <= t_end)
    idx = night_df.index[mask]

    if len(idx) < 60:  # need at least 1 hour
        return None

    # Time in hours from dome-open (negative = before open)
    t_hours = (idx - t_open).total_seconds() / 3600.0

    # Compute T_floor: average ESS 113 (M1M3 level) in the 1h before dome-open
    # Falls back to ESS 112 (T_in) if ESS 113 not available
    pre_mask = (night_df.index >= t_start) & (night_df.index < t_open)
    if "ess113_temp_mean" in night_df.columns:
        ess113_raw = night_df["ess113_temp_mean"].values
        ess113_pre = ess113_raw[pre_mask]
        T_floor = np.nanmean(ess113_pre) if np.any(np.isfinite(ess113_pre)) else T_in_smooth[mask][0]
    else:
        T_in_pre = T_in_smooth[pre_mask]
        T_floor = np.nanmean(T_in_pre) if np.any(np.isfinite(T_in_pre)) else T_in_smooth[mask][0]

    # Ground-level temperature: ESS 113 (M1M3 level air)
    T_ground = None
    if "ess113_temp_mean" in night_df.columns:
        raw = night_df["ess113_temp_mean"].values
        if np.isfinite(raw[mask]).sum() > 30:
            T_ground = smooth_temperatures(raw, window_min=15)[mask]

    # Get wind speed if available
    wind = None
    if "wind_speed_mean" in night_df.columns:
        wind = night_df.loc[idx, "wind_speed_mean"].values

    # Get night_date
    night_date = None
    if "night_date" in night_df.columns:
        night_date = night_df["night_date"].iloc[0]

    return {
        "t": t_hours,
        "T_in": T_in_smooth[mask],
        "T_out": T_out_smooth[mask],
        "T_in_raw": T_in_raw[mask],
        "T_out_raw": T_out_raw[mask],
        "T_ground": T_ground,
        "t_open": t_open,
        "T_floor": T_floor,
        "dome_hours": dome_hours,
        "wind_speed": wind,
        "night_date": night_date,
        "index": idx,
    }


# --------------------------------------------------------------------------- #
# ODE Models
# --------------------------------------------------------------------------- #

def _interp_T_out(t, t_data, T_out_data):
    """Interpolate outside temperature at arbitrary time."""
    return np.interp(t, t_data, T_out_data)


def _interp_dTout_dt(t, t_data, T_out_data):
    """Interpolate dT_out/dt at arbitrary time."""
    dt = np.gradient(T_out_data, t_data)
    return np.interp(t, t_data, dt)


def model0_exponential(t, T_in0, T_out0, tau):
    """
    Model 0: Simple exponential decay.
    ΔT(t) = ΔT_0 * exp(-t/τ), then T_in = T_out0 + ΔT(t).
    Assumes constant T_out = T_out0. Expected to fail.
    """
    dT0 = T_in0 - T_out0
    return T_out0 + dT0 * np.exp(-t / tau)


def simulate_ode(t_data, T_out_data, T_in0, params, model_type="basic"):
    """
    Forward-integrate an ODE model against actual T_out time series.

    Parameters
    ----------
    t_data : array
        Time in hours.
    T_out_data : array
        Outside temperature at each time step.
    T_in0 : float
        Initial inside temperature.
    params : dict
        Model parameters (tau, k, alpha, T_floor depending on model_type).
    model_type : str
        One of "basic", "derivative", "deep_mass".

    Returns
    -------
    np.ndarray
        Predicted inside temperature at each time step.
    """
    tau = params["tau"]

    def rhs_basic(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        return (T_out - T_in[0]) / tau

    def rhs_derivative(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        dTout = _interp_dTout_dt(t, t_data, T_out_data)
        k = params["k"]
        return (T_out - T_in[0]) / tau + k * dTout

    def rhs_offset(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        dTout = _interp_dTout_dt(t, t_data, T_out_data)
        k = params["k"]
        dT_off = params["dT_offset"]
        return (T_out + dT_off - T_in[0]) / tau + k * dTout

    def rhs_deep_mass(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        alpha = params["alpha"]
        T_floor = params["T_floor"]
        T_target = alpha * T_out + (1 - alpha) * T_floor
        return (T_target - T_in[0]) / tau

    def rhs_full(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        dTout = _interp_dTout_dt(t, t_data, T_out_data)
        alpha = params["alpha"]
        T_floor = params["T_floor"]
        k = params["k"]
        T_target = alpha * T_out + (1 - alpha) * T_floor
        return (T_target - T_in[0]) / tau + k * dTout

    def rhs_buoyancy(t, T_in):
        T_out = _interp_T_out(t, t_data, T_out_data)
        alpha = params["alpha"]
        T_floor = params["T_floor"]
        C_buoy = params["C_buoy"]
        T_target = alpha * T_out + (1 - alpha) * T_floor
        delta_T = abs(T_out - T_in[0])
        # Dynamic tau: 1/tau_dyn = C_buoy * sqrt(|delta_T| / T_floor)
        exchange_rate = C_buoy * np.sqrt(max(delta_T / T_floor, 1e-8))
        return exchange_rate * (T_target - T_in[0])

    def rhs_ground_no_deriv(t, T_in):
        """
        Ground-coupled without derivative: uses measured T_ground(t).
        dT_in/dt = [α*T_out(t) + (1-α)*T_ground(t) - T_in] / τ
        """
        T_out = _interp_T_out(t, t_data, T_out_data)
        T_ground_data = params["T_ground_data"]
        T_gnd = np.interp(t, t_data, T_ground_data)
        alpha = params["alpha"]
        T_target = alpha * T_out + (1 - alpha) * T_gnd
        return (T_target - T_in[0]) / tau

    def rhs_ground_coupled(t, T_in):
        """
        Ground-coupled + derivative: uses measured T_ground(t) + wind correction.
        dT_in/dt = [α*T_out(t) + (1-α)*T_ground(t) - T_in] / τ + k*dT_out/dt
        """
        T_out = _interp_T_out(t, t_data, T_out_data)
        dTout = _interp_dTout_dt(t, t_data, T_out_data)
        T_ground_data = params["T_ground_data"]
        T_gnd = np.interp(t, t_data, T_ground_data)
        alpha = params["alpha"]
        k = params["k"]
        T_target = alpha * T_out + (1 - alpha) * T_gnd
        return (T_target - T_in[0]) / tau + k * dTout

    def rhs_ramped(t, T_in):
        """
        Ramped mixing: exchange efficiency builds up as convective mixing
        propagates through the dome volume.
        f(t) = 1 - exp(-t/τ_mix)
        dT_in/dt = [f(t)/τ] * [α*T_out + (1-α)*T_ground - T_in] + k*dT_out/dt
        """
        T_out = _interp_T_out(t, t_data, T_out_data)
        dTout = _interp_dTout_dt(t, t_data, T_out_data)
        T_ground_data = params["T_ground_data"]
        T_gnd = np.interp(t, t_data, T_ground_data)
        alpha = params["alpha"]
        k = params["k"]
        tau_mix = params["tau_mix"]
        # Mixing efficiency ramps from 0 to 1
        t_since_open = max(t - t_data[0], 0)
        f = 1.0 - np.exp(-t_since_open / max(tau_mix, 1e-6))
        T_target = alpha * T_out + (1 - alpha) * T_gnd
        return (f / tau) * (T_target - T_in[0]) + k * dTout

    rhs = {"basic": rhs_basic, "derivative": rhs_derivative,
           "offset": rhs_offset,
           "deep_mass": rhs_deep_mass, "full": rhs_full,
           "buoyancy": rhs_buoyancy,
           "ground_no_deriv": rhs_ground_no_deriv,
           "ground_coupled": rhs_ground_coupled,
           "ramped": rhs_ramped}[model_type]

    sol = solve_ivp(
        rhs,
        t_span=(t_data[0], t_data[-1]),
        y0=[T_in0],
        t_eval=t_data,
        method="RK45",
        max_step=1.0 / 60.0,  # max 1-minute steps
    )

    if sol.success:
        return sol.y[0]
    else:
        return np.full_like(t_data, np.nan)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def fit_model0(t, T_in_obs, T_out_obs, **kwargs):
    """Fit simple exponential decay. Returns (tau, rmse)."""
    T_in0 = T_in_obs[0]
    T_out0 = T_out_obs[0]

    def residuals(p):
        tau = p[0]
        pred = model0_exponential(t, T_in0, T_out0, tau)
        return pred - T_in_obs

    result = least_squares(residuals, x0=[2.0], bounds=([0.1], [24.0]))
    tau = result.x[0]
    pred = model0_exponential(t, T_in0, T_out0, tau)
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "rmse": rmse, "pred": pred, "model": "exponential"}


def fit_model1(t, T_in_obs, T_out_obs, **kwargs):
    """Fit basic ODE (moving target). Returns (tau, rmse)."""
    T_in0 = T_in_obs[0]

    def residuals(p):
        tau = p[0]
        pred = simulate_ode(t, T_out_obs, T_in0, {"tau": tau}, "basic")
        return pred - T_in_obs

    result = least_squares(residuals, x0=[2.0], bounds=([0.1], [24.0]))
    tau = result.x[0]
    pred = simulate_ode(t, T_out_obs, T_in0, {"tau": tau}, "basic")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "rmse": rmse, "pred": pred, "model": "basic_ode"}


def fit_model2(t, T_in_obs, T_out_obs, **kwargs):
    """
    Fit forced ventilation model with temperature offset.
    dT_in/dt = (T_out + ΔT_offset - T_in) / τ + k * dT_out/dt

    The offset captures the persistent warm bias from structural mass.
    """
    T_in0 = T_in_obs[0]

    def residuals(p):
        tau, k, dT_off = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": tau, "k": k, "dT_offset": dT_off}, "offset")
        return pred - T_in_obs

    result = least_squares(residuals, x0=[2.0, 0.3, 0.5],
                           bounds=([0.1, -1.0, -5.0], [24.0, 2.0, 5.0]))
    tau, k, dT_off = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": tau, "k": k, "dT_offset": dT_off}, "offset")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "k": k, "dT_offset": dT_off, "rmse": rmse,
            "pred": pred, "model": "forced_ventilation"}


def fit_model3(t, T_in_obs, T_out_obs, T_ground=None, **kwargs):
    """
    Fit stratified model (no derivative) with measured ESS 113.
    dT_in/dt = [α*T_out(t) + (1-α)*T_ess113(t) - T_in] / τ

    Parameters fitted: τ, α  (T_ground = ESS 113 time series, not fitted)
    """
    T_in0 = T_in_obs[0]
    if T_ground is None:
        return {"tau": np.nan, "alpha": np.nan, "rmse": np.nan,
                "pred": np.full_like(t, np.nan), "model": "stratified_no_deriv"}

    T_gnd = pd.Series(T_ground).interpolate(limit_direction="both").values

    def residuals(p):
        tau, alpha = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": tau, "alpha": alpha, "T_ground_data": T_gnd},
                            "ground_no_deriv")
        return pred - T_in_obs

    result = least_squares(
        residuals,
        x0=[3.0, 0.5],
        bounds=([0.1, 0.01], [24.0, 0.99]),
    )
    tau, alpha = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": tau, "alpha": alpha, "T_ground_data": T_gnd},
                        "ground_no_deriv")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "alpha": alpha, "rmse": rmse,
            "pred": pred, "model": "stratified_no_deriv"}


def fit_model4(t, T_in_obs, T_out_obs, T_floor=None):
    """
    Fit full model: deep-mass + derivative coupling, with fixed T_floor.
    dT_in/dt = (α*T_out + (1-α)*T_floor - T_in) / τ + k * dT_out/dt

    Parameters fitted: τ, α, k  (T_floor fixed from pre-open data)
    """
    T_in0 = T_in_obs[0]
    if T_floor is None:
        T_floor = T_in0

    def residuals(p):
        tau, alpha, k = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": tau, "alpha": alpha,
                             "T_floor": T_floor, "k": k},
                            "full")
        return pred - T_in_obs

    result = least_squares(
        residuals,
        x0=[3.0, 0.4, 0.3],
        bounds=([0.1, 0.01, -1.0],
                [24.0, 0.99, 2.0]),
    )
    tau, alpha, k = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": tau, "alpha": alpha,
                         "T_floor": T_floor, "k": k},
                        "full")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "alpha": alpha, "T_floor": T_floor, "k": k,
            "rmse": rmse, "pred": pred, "model": "full"}


def fit_model5(t, T_in_obs, T_out_obs, T_floor=None):
    """
    Fit buoyancy-driven model with deep-mass coupling, fixed T_floor.
    dT_in/dt = C_buoy * sqrt(|T_out - T_in|) * (α*T_out + (1-α)*T_floor - T_in)

    Parameters fitted: C_buoy, α  (T_floor fixed from pre-open data)
    No static tau — the exchange rate is entirely driven by ΔT.
    """
    T_in0 = T_in_obs[0]
    if T_floor is None:
        T_floor = T_in0

    def residuals(p):
        C_buoy, alpha = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": 1.0, "C_buoy": C_buoy, "alpha": alpha,
                             "T_floor": T_floor},
                            "buoyancy")
        return pred - T_in_obs

    result = least_squares(
        residuals,
        x0=[0.5, 0.5],
        bounds=([0.01, 0.01], [10.0, 0.99]),
    )
    C_buoy, alpha = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": 1.0, "C_buoy": C_buoy, "alpha": alpha,
                         "T_floor": T_floor},
                        "buoyancy")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"C_buoy": C_buoy, "alpha": alpha, "T_floor": T_floor,
            "rmse": rmse, "pred": pred, "model": "buoyancy"}


def fit_model6(t, T_in_obs, T_out_obs, T_ground=None, **kwargs):
    """
    Fit ground-coupled model: measured T_ground(t) replaces constant T_floor.
    dT_in/dt = [α*T_out(t) + (1-α)*T_ground(t) - T_in] / τ + k*dT_out/dt

    Parameters fitted: τ, α, k  (T_ground is measured data, not fitted)
    """
    T_in0 = T_in_obs[0]
    if T_ground is None:
        return {"tau": np.nan, "alpha": np.nan, "k": np.nan,
                "rmse": np.nan, "pred": np.full_like(t, np.nan),
                "model": "ground_coupled"}

    # Ensure T_ground has no NaNs for interpolation
    T_ground_clean = pd.Series(T_ground).interpolate(limit_direction="both").values

    def residuals(p):
        tau, alpha, k = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": tau, "alpha": alpha, "k": k,
                             "T_ground_data": T_ground_clean},
                            "ground_coupled")
        return pred - T_in_obs

    result = least_squares(
        residuals,
        x0=[3.0, 0.5, 0.2],
        bounds=([0.1, 0.01, -1.0],
                [24.0, 0.99, 2.0]),
    )
    tau, alpha, k = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": tau, "alpha": alpha, "k": k,
                         "T_ground_data": T_ground_clean},
                        "ground_coupled")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "alpha": alpha, "k": k,
            "rmse": rmse, "pred": pred, "model": "ground_coupled"}


def fit_model7(t, T_in_obs, T_out_obs, T_ground=None, **kwargs):
    """
    Ramped mixing model: exchange efficiency builds up over tau_mix.
    f(t) = 1 - exp(-t/τ_mix)
    dT_in/dt = [f(t)/τ] * [α*T_out + (1-α)*T_ground - T_in] + k*dT_out/dt

    Parameters fitted: τ, α, k, τ_mix
    Two time constants: τ (equilibration) and τ_mix (mixing propagation delay).
    """
    T_in0 = T_in_obs[0]
    if T_ground is None:
        return {"tau": np.nan, "alpha": np.nan, "k": np.nan, "tau_mix": np.nan,
                "rmse": np.nan, "pred": np.full_like(t, np.nan), "model": "ramped"}

    T_gnd = pd.Series(T_ground).interpolate(limit_direction="both").values

    def residuals(p):
        tau, alpha, k, tau_mix = p
        pred = simulate_ode(t, T_out_obs, T_in0,
                            {"tau": tau, "alpha": alpha, "k": k,
                             "tau_mix": tau_mix, "T_ground_data": T_gnd},
                            "ramped")
        return pred - T_in_obs

    result = least_squares(
        residuals,
        x0=[0.5, 0.5, 0.1, 0.2],
        bounds=([0.01, 0.01, -1.0, 0.01],
                [12.0, 0.99, 2.0, 3.0]),
    )
    tau, alpha, k, tau_mix = result.x
    pred = simulate_ode(t, T_out_obs, T_in0,
                        {"tau": tau, "alpha": alpha, "k": k,
                         "tau_mix": tau_mix, "T_ground_data": T_gnd},
                        "ramped")
    rmse = np.sqrt(np.mean((pred - T_in_obs) ** 2))
    return {"tau": tau, "alpha": alpha, "k": k, "tau_mix": tau_mix,
            "rmse": rmse, "pred": pred, "model": "ramped"}


def compute_bic(rmse, n, k):
    """Bayesian Information Criterion: BIC = n*ln(σ²) + k*ln(n)."""
    if rmse <= 0 or n <= 0:
        return np.inf
    return n * np.log(rmse ** 2) + k * np.log(n)


def fit_all_models(night_data):
    """
    Fit all 4 models to a single night.

    Parameters
    ----------
    night_data : dict
        Output from prepare_night().

    Returns
    -------
    dict
        Per-model results + metadata.
    """
    t = night_data["t"]
    T_in = night_data["T_in"]
    T_out = night_data["T_out"]

    # Remove NaN from fitting
    valid = np.isfinite(T_in) & np.isfinite(T_out)
    if valid.sum() < 60:
        return None

    t_v = t[valid]
    Ti_v = T_in[valid]
    To_v = T_out[valid]
    n = len(t_v)

    T_floor = night_data.get("T_floor", Ti_v[0])

    results = {"night_date": night_data["night_date"], "n_points": n,
                "dome_hours": night_data.get("dome_hours", np.nan),
                "T_floor": T_floor}

    # Wind stats
    if night_data["wind_speed"] is not None:
        ws = night_data["wind_speed"]
        results["wind_speed_mean"] = np.nanmean(ws)
        results["wind_speed_std"] = np.nanstd(ws)

    # Initial delta_T
    results["delta_T_initial"] = Ti_v[0] - To_v[0]

    # Get T_ground for M6 (may be None if no ground sensor data)
    T_ground = night_data.get("T_ground")
    T_ground_v = T_ground[valid] if T_ground is not None else None

    # Fit each model
    # M0-M2: single-source (outside air only)
    # M3: stratified (ESS 113 ground coupling, no derivative)
    # M6: stratified + wind (ESS 113 ground coupling + derivative)
    for name, fit_func, n_params, extra_kw in [
        ("m0", fit_model0, 1, {}),
        ("m1", fit_model1, 1, {}),
        ("m2", fit_model2, 3, {}),
        ("m3", fit_model3, 2, {"T_ground": T_ground_v}),
        ("m6", fit_model6, 3, {"T_ground": T_ground_v}),
    ]:
        try:
            r = fit_func(t_v, Ti_v, To_v, **extra_kw)
            results[f"{name}_rmse"] = r["rmse"]
            results[f"{name}_bic"] = compute_bic(r["rmse"], n, n_params)
            if "tau" in r:
                results[f"{name}_tau"] = r["tau"]
            if "k" in r:
                results[f"{name}_k"] = r["k"]
            if "alpha" in r:
                results[f"{name}_alpha"] = r["alpha"]
            if "T_floor" in r:
                results[f"{name}_T_floor"] = r["T_floor"]
            if "C_buoy" in r:
                results[f"{name}_C_buoy"] = r["C_buoy"]
            if "dT_offset" in r:
                results[f"{name}_dT_offset"] = r["dT_offset"]
            results[f"{name}_pred"] = r["pred"]
        except Exception as e:
            results[f"{name}_rmse"] = np.nan
            results[f"{name}_bic"] = np.nan
            results[f"{name}_tau"] = np.nan

    return results
