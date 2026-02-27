#!/usr/bin/env python3
"""
Analyze dome thermalization from individual open/close transient events.

Instead of fitting one tau per night, this script:
1. Detects all dome open/close transitions across all nights
2. Extracts the step response for each event
3. Detects the thermalization plateau (when ΔT stabilizes)
4. Fits tau from each transient individually
5. Builds a catalog of transient events with tau, wind, initial ΔT

This gives much cleaner tau estimates because:
- Each event has a clear start time
- Short duration (minutes to hours) avoids the moving-target problem
- Plateau detection avoids fitting noise
- Multiple events per night on intermittent nights

Usage:
    python dome_env/transient_analysis.py
    python dome_env/transient_analysis.py --min-duration 10 --max-duration 180
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from dome_thermal import (
    load_thermal_data, smooth_temperatures, simulate_ode,
)

FIGURES_DIR = Path(__file__).parent / "figures" / "transients"
RESULTS_PATH = Path(__file__).parent / "transient_events.csv"


def detect_dome_transitions(night_df):
    """
    Find all dome open and close transitions in a night.

    Returns list of dicts with:
    - t_start: transition time
    - event_type: 'open' or 'close'
    - t_end: next opposite transition (or end of data)
    - duration_min: duration in minutes
    """
    if "dome_open" not in night_df.columns:
        return []

    dome = night_df["dome_open"].fillna(False)
    if hasattr(dome.iloc[0], '__bool__'):
        dome = dome.astype(bool)
    else:
        dome = dome.map(lambda x: x == True or x == 'True')

    transitions = dome.astype(int).diff().fillna(0)
    events = []

    openings = night_df.index[transitions == 1].tolist()
    closings = night_df.index[transitions == -1].tolist()

    # Build list of all transitions in time order
    all_trans = [(t, "open") for t in openings] + [(t, "close") for t in closings]
    all_trans.sort(key=lambda x: x[0])

    for i, (t_start, event_type) in enumerate(all_trans):
        # End time = next opposite transition, or end of data
        if i + 1 < len(all_trans):
            t_end = all_trans[i + 1][0]
        else:
            t_end = night_df.index[-1]

        duration_min = (t_end - t_start).total_seconds() / 60.0

        events.append({
            "t_start": t_start,
            "t_end": t_end,
            "event_type": event_type,
            "duration_min": duration_min,
        })

    return events


def detect_onset(delta_T, t_hours, window=15):
    """
    Detect the thermalization onset: max|ΔT| where d²(ΔT)/dt² crosses zero.

    This is the inflection point of the ΔT curve — the moment the cooling
    rate is at its maximum, after which the curve decelerates toward plateau.
    Typically ~30 min after dome opening.

    Parameters
    ----------
    delta_T : array
        Temperature differential (T_in - T_out).
    t_hours : array
        Time in hours (t=0 at dome open).
    window : int
        Smoothing window for derivatives (minutes).

    Returns
    -------
    int
        Index of onset.
    """
    dt = np.gradient(t_hours)
    dt[dt == 0] = 1.0 / 60.0

    # Compute acceleration d²(ΔT)/dt²
    d1 = pd.Series(np.gradient(delta_T) / dt).rolling(window=window, center=True).mean().values
    d2 = pd.Series(np.gradient(d1) / dt).rolling(window=window, center=True).mean().values

    # Search within first 1.5h after dome open
    valid = np.isfinite(d2) & (t_hours >= 0) & (t_hours <= 1.5)
    if not valid.any():
        return 0

    indices = np.where(valid)[0]

    # Find zero-crossings of acceleration
    for i in range(1, len(indices)):
        j0, j1 = indices[i - 1], indices[i]
        if d2[j0] * d2[j1] < 0:  # sign change
            # Pick the one with max |ΔT|
            if abs(delta_T[j0]) >= abs(delta_T[j1]):
                return j0
            else:
                return j1

    # Fallback: max |ΔT| in the window
    return indices[np.argmax(np.abs(delta_T[indices]))]


def detect_plateau(delta_T, t_min, window_min=60, noise_threshold=0.15):
    """
    Detect when ΔT reaches a plateau (thermalization complete).

    Plateau = low noise (rolling std < 0.15C) AND flat (rolling slope < 0.1 C/h),
    sustained for at least 10 consecutive minutes.

    Parameters
    ----------
    delta_T : array
        Temperature differential (T_in - T_out) at 1-min resolution.
    t_min : array
        Time in minutes from event start.
    window_min : int
        Rolling window size in minutes.
    noise_threshold : float
        Std threshold in C to declare plateau.

    Returns
    -------
    int or None
        Index where plateau starts, or None if not detected.
    """
    if len(delta_T) < window_min + 10:
        return None

    series = pd.Series(delta_T)
    rolling_std = series.rolling(window=window_min, center=True).std()

    rolling_slope = series.rolling(window=window_min, center=True).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 2 else np.nan,
        raw=True
    )

    plateau_mask = (rolling_std < noise_threshold) & (np.abs(rolling_slope) < 0.1)

    # Find first sustained plateau (at least 10 consecutive minutes)
    if not plateau_mask.any():
        return None

    consecutive = 0
    for i in range(len(plateau_mask)):
        if plateau_mask.iloc[i]:
            consecutive += 1
            if consecutive >= 10:
                return i - 9
        else:
            consecutive = 0

    return None


def _solve_deltaT(t_data, dTout_dt, dT0, params, model_type):
    """
    Solve ODE in the ΔT frame.

    Forced Ventilation: dΔT/dt = -ΔT/τ - dT_out/dt
    With offset:        dΔT/dt = -(ΔT - ΔT_eq)/τ - dT_out/dt
    Stratified:         dΔT/dt = -(ΔT - (1-α)·ΔT_ground)/τ - dT_out/dt
                        where ΔT_ground = T_ground - T_out
    """
    from scipy.integrate import solve_ivp

    tau = params["tau"]

    def rhs_forced(t, dT):
        dTout = np.interp(t, t_data, dTout_dt)
        return -dT[0] / tau - dTout

    def rhs_offset(t, dT):
        dTout = np.interp(t, t_data, dTout_dt)
        dT_eq = params["dT_eq"]
        return -(dT[0] - dT_eq) / tau - dTout

    def rhs_stratified(t, dT):
        dTout = np.interp(t, t_data, dTout_dt)
        alpha = params["alpha"]
        dT_ground_data = params["dT_ground_data"]
        dT_gnd = np.interp(t, t_data, dT_ground_data)
        return -(dT[0] - (1 - alpha) * dT_gnd) / tau - dTout

    rhs = {"forced": rhs_forced, "offset": rhs_offset,
           "stratified": rhs_stratified}[model_type]

    sol = solve_ivp(rhs, t_span=(t_data[0], t_data[-1]), y0=[dT0],
                    t_eval=t_data, method="RK45", max_step=1.0 / 60.0)

    if sol.success:
        return sol.y[0]
    return np.full_like(t_data, np.nan)


def fit_transient(t_hours, T_in, T_out, T_mirror=None):
    """
    Fit transient in the ΔT frame.

    Models:
    - Forced Ventilation: dΔT/dt = -ΔT/τ - dT_out/dt  (1 param: τ)
    - With offset: dΔT/dt = -(ΔT - ΔT_eq)/τ - dT_out/dt  (2 params: τ, ΔT_eq)
    - Stratified: dΔT/dt = -(ΔT - (1-α)·ΔT_ground)/τ - dT_out/dt  (2 params: τ, α)
    """
    delta_T = T_in - T_out
    dT0 = delta_T[0]

    # Compute dT_out/dt (smoothed)
    dt = np.gradient(t_hours)
    dt[dt == 0] = 1.0 / 60.0
    dTout_dt = pd.Series(np.gradient(T_out) / dt).rolling(15, center=True).mean().fillna(0).values

    results = {}

    # M2a: Pure forced ventilation (1 param: τ)
    try:
        def res_m2a(p):
            tau = p[0]
            pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0, {"tau": tau}, "forced")
            return pred_dT - delta_T

        r = least_squares(res_m2a, x0=[0.5], bounds=([0.01], [12.0]))
        tau = r.x[0]
        pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0, {"tau": tau}, "forced")
        rmse = np.sqrt(np.mean((pred_dT - delta_T) ** 2))
        results["m2_tau"] = tau
        results["m2_rmse"] = rmse
        results["m2_pred"] = pred_dT + T_out  # convert back to T_in for plotting
        results["m2_pred_dT"] = pred_dT
    except Exception:
        results["m2_tau"] = np.nan

    # M2b: Forced ventilation with equilibrium offset (2 params: τ, ΔT_eq)
    try:
        def res_m2b(p):
            tau, dT_eq = p
            pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0,
                                     {"tau": tau, "dT_eq": dT_eq}, "offset")
            return pred_dT - delta_T

        r = least_squares(res_m2b, x0=[0.5, 0.0],
                          bounds=([0.01, -5.0], [12.0, 5.0]))
        tau, dT_eq = r.x
        pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0,
                                 {"tau": tau, "dT_eq": dT_eq}, "offset")
        rmse = np.sqrt(np.mean((pred_dT - delta_T) ** 2))
        results["m2b_tau"] = tau
        results["m2b_dT_eq"] = dT_eq
        results["m2b_rmse"] = rmse
        results["m2b_pred"] = pred_dT + T_out
        results["m2b_pred_dT"] = pred_dT
    except Exception:
        results["m2b_tau"] = np.nan

    # M6: Stratified (2 params: τ, α) — needs T_mirror
    if T_mirror is not None and np.isfinite(T_mirror).sum() > 10:
        try:
            T_mirror_clean = pd.Series(T_mirror).interpolate(limit_direction="both").values
            dT_ground = T_mirror_clean - T_out  # ΔT_ground = T_ground - T_out

            def res_m6(p):
                tau, alpha = p
                pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0,
                                         {"tau": tau, "alpha": alpha,
                                          "dT_ground_data": dT_ground}, "stratified")
                return pred_dT - delta_T

            r = least_squares(res_m6, x0=[0.5, 0.5],
                              bounds=([0.01, 0.01], [12.0, 0.99]))
            tau, alpha = r.x
            pred_dT = _solve_deltaT(t_hours, dTout_dt, dT0,
                                     {"tau": tau, "alpha": alpha,
                                      "dT_ground_data": dT_ground}, "stratified")
            rmse = np.sqrt(np.mean((pred_dT - delta_T) ** 2))
            results["m6_tau"] = tau
            results["m6_alpha"] = alpha
            results["m6_rmse"] = rmse
            results["m6_pred"] = pred_dT + T_out
            results["m6_pred_dT"] = pred_dT
        except Exception:
            results["m6_tau"] = np.nan

    return results


def process_event(night_df, event, min_duration=10, max_duration=None,
                  fit_window_min=None):
    """Process a single dome open/close transient event."""
    duration = event["duration_min"]
    if duration < min_duration:
        return None
    if max_duration is not None and duration > max_duration:
        return None

    t_start = event["t_start"]
    t_end = event["t_end"]

    # Extract data window: 30 min before event to end
    pre_start = t_start - pd.Timedelta(minutes=30)
    mask = (night_df.index >= pre_start) & (night_df.index <= t_end)
    segment = night_df.loc[mask]

    if len(segment) < 15:
        return None

    # Get temperatures
    T_in_raw = segment["inside_m2_temp_mean"].values
    T_out_raw = segment["outside_temp_mean"].values

    if np.isfinite(T_in_raw).sum() < 10 or np.isfinite(T_out_raw).sum() < 10:
        return None

    # Smoothing: 15-min for inside (and ESS 113), 61-min for outside
    # For plots we also keep a 15-min outside version
    # All temperatures smoothed to 1h for consistency
    T_in_1h = smooth_temperatures(T_in_raw, window_min=61, polyorder=2)
    T_out_1h = smooth_temperatures(T_out_raw, window_min=61, polyorder=2)

    T_in = T_in_1h
    T_out = T_out_1h
    # For plotting: keep references consistent
    T_out_15 = T_out_1h  # no separate 15min version
    T_in_15 = T_in_1h

    # Time in hours from event start
    t_hours = (segment.index - t_start).total_seconds() / 3600.0

    # Valid mask
    valid = np.isfinite(T_in) & np.isfinite(T_out)
    if valid.sum() < 10:
        return None

    delta_T = T_in[valid] - T_out[valid]
    t_min = (segment.index[valid] - t_start).total_seconds() / 60.0  # minutes from dome open

    # Detect onset: max|ΔT| where acceleration crosses zero (~30min after open)
    onset_idx = detect_onset(delta_T, t_hours[valid])

    # Plateau detection (skipped when using fixed fit window)
    if fit_window_min is not None:
        plateau_idx = None
    else:
        delta_T_raw = T_in_raw[valid] - T_out_raw[valid]
        plateau_idx = detect_plateau(delta_T_raw, t_min)

    # ESS 113 as ground-level proxy
    T_mirror = None
    for col in ["ess113_temp_mean", "inside_m1m3_temp_mean"]:
        if col in segment.columns:
            T_mirror_raw = segment[col].values
            if np.isfinite(T_mirror_raw).sum() > 10:
                T_mirror = smooth_temperatures(T_mirror_raw, window_min=61, polyorder=2)
            break

    # Fit from dome open (t=0)
    t0_idx = np.searchsorted(t_hours[valid], 0)
    fit_start = t0_idx

    if fit_window_min is not None:
        # Fixed time window: fit from t=0 to t=fit_window_min
        fit_end_idx = np.searchsorted(t_min, fit_window_min)
        fit_end = min(fit_end_idx, valid.sum())
    elif plateau_idx is not None and plateau_idx > fit_start + 5:
        fit_end = plateau_idx
    else:
        fit_end = valid.sum()

    tv = t_hours[valid][fit_start:fit_end]
    Ti = T_in[valid][fit_start:fit_end]
    To = T_out[valid][fit_start:fit_end]
    Tm = T_mirror[valid][fit_start:fit_end] if T_mirror is not None else None

    if len(tv) < 10:
        return None

    fit = fit_transient(tv, Ti, To, Tm)

    # Wind speed
    wind_mean = np.nan
    if "wind_speed_mean" in segment.columns:
        wind_mean = segment["wind_speed_mean"].mean()

    # Night date
    night_date = segment["night_date"].iloc[0] if "night_date" in segment.columns else None

    return {
        "night_date": night_date,
        "t_start": str(t_start),
        "event_type": event["event_type"],
        "duration_min": duration,
        "onset_min": t_min[onset_idx] if onset_idx < len(t_min) else np.nan,
        "delta_T_at_onset": delta_T[onset_idx],
        "delta_T_final": delta_T[-1],
        "delta_T_at_plateau": delta_T[plateau_idx] if plateau_idx is not None else delta_T[-1],
        "plateau_min": t_min[plateau_idx] if plateau_idx is not None else np.nan,
        "wind_speed_mean": wind_mean,
        **{k: v for k, v in fit.items() if not k.endswith("_pred")},
        # Store for plotting
        "_t_hours": t_hours[valid],
        "_T_in": T_in[valid],
        "_T_out_1h": T_out[valid],
        "_T_in_raw": T_in_raw[valid],
        "_T_out_15": T_out_15[valid],
        "_delta_T": delta_T,
        "_t_min": t_min,
        "_onset_idx": onset_idx,
        "_plateau_idx": plateau_idx,
        "_m2_pred": fit.get("m2_pred"),
        "_m2b_pred": fit.get("m2b_pred"),
        "_m6_pred": fit.get("m6_pred"),
        "_fit_start": fit_start,
        "_fit_end": fit_end,
    }


def plot_transient(event_data, output_dir):
    """Plot a single transient event with both smoothing levels."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), height_ratios=[2.5, 1, 1],
                              sharex=True, gridspec_kw={"hspace": 0.08})

    t_h = event_data["_t_hours"]
    Ti = event_data["_T_in"]
    To_1h = event_data["_T_out_1h"]
    To_15 = event_data["_T_out_15"]
    dT = event_data["_delta_T"]
    t_min_arr = event_data["_t_min"]
    plateau_idx = event_data["_plateau_idx"]
    fit_start = event_data.get("_fit_start", 0)
    fit_end = event_data["_fit_end"]

    t_plot = t_min_arr  # minutes from dome open

    # X-axis: -30min to 2h
    xlim = (max(t_min_arr[0], -30), min(120, t_min_arr[-1]))

    # Extrapolation range: from fit_start to 2h (or end of data)
    ext_end_idx = np.searchsorted(t_min_arr, 120)
    ext_end_idx = min(ext_end_idx, len(t_min_arr))
    ext_t_h = t_h[fit_start:ext_end_idx]
    ext_t_min = t_min_arr[fit_start:ext_end_idx]
    ext_To = To_1h[fit_start:ext_end_idx]
    ext_Ti = Ti[fit_start:ext_end_idx]

    # Compute dT_out/dt for extrapolation
    dt_ext = np.gradient(ext_t_h)
    dt_ext[dt_ext == 0] = 1.0 / 60.0
    dTout_ext = pd.Series(np.gradient(ext_To) / dt_ext).rolling(15, center=True).mean().fillna(0).values

    # ΔT at fit start
    dT0 = Ti[fit_start] - To_1h[fit_start]

    # Fit window markers
    fit_end_min = t_min_arr[min(fit_end, len(t_min_arr) - 1)]

    # Panel 1: Temperatures + model extrapolations
    ax = axes[0]
    ax.plot(t_plot, event_data["_T_in_raw"], color="lightcoral", alpha=0.2, linewidth=0.5)
    ax.plot(t_plot, Ti, color="tab:red", linewidth=2, label="$T_{in}$ (15min)")
    ax.plot(t_plot, To_15, color="lightskyblue", linewidth=1.2, alpha=0.7,
            label="$T_{out}$ (15min)")
    ax.plot(t_plot, To_1h, color="tab:blue", linewidth=2, label="$T_{out}$ (1h)")

    # Re-solve each model on the full 2h range for extrapolation
    tau_m2 = event_data.get("m2_tau", np.nan)
    if np.isfinite(tau_m2):
        pred_dT = _solve_deltaT(ext_t_h, dTout_ext, dT0, {"tau": tau_m2}, "forced")
        pred_Tin = pred_dT + ext_To
        ax.plot(ext_t_min[:len(pred_Tin)], pred_Tin, "--", color="#6baed6", linewidth=2,
                label=f"Forced: $\\tau$={tau_m2*60:.0f}min")

    tau_m2b = event_data.get("m2b_tau", np.nan)
    dT_eq = event_data.get("m2b_dT_eq", 0)
    if np.isfinite(tau_m2b):
        pred_dT = _solve_deltaT(ext_t_h, dTout_ext, dT0,
                                 {"tau": tau_m2b, "dT_eq": dT_eq}, "offset")
        pred_Tin = pred_dT + ext_To
        ax.plot(ext_t_min[:len(pred_Tin)], pred_Tin, "--", color="#fc8d59", linewidth=2,
                label=f"Offset: $\\tau$={tau_m2b*60:.0f}min, $\\Delta T_{{eq}}$={dT_eq:.1f}")

    tau_m6 = event_data.get("m6_tau", np.nan)
    alpha_m6 = event_data.get("m6_alpha", np.nan)
    if np.isfinite(tau_m6):
        # Need ESS 113 for stratified extrapolation
        for col in ["ess113_temp_mean", "inside_m1m3_temp_mean"]:
            if col in str(event_data.get("_m6_pred", "")):
                break
        # Approximate: use T_in as proxy for T_ground in extrapolation
        dT_gnd = ext_Ti - ext_To
        pred_dT = _solve_deltaT(ext_t_h, dTout_ext, dT0,
                                 {"tau": tau_m6, "alpha": alpha_m6,
                                  "dT_ground_data": dT_gnd}, "stratified")
        pred_Tin = pred_dT + ext_To
        ax.plot(ext_t_min[:len(pred_Tin)], pred_Tin, "--", color="#2171b5", linewidth=2,
                label=f"Stratified: $\\tau$={tau_m6*60:.0f}min, $\\alpha$={alpha_m6:.2f}")

    # Mark fit window end
    ax.axvline(fit_end_min, color="gray", linewidth=1.5, linestyle=":",
               alpha=0.5, label=f"Fit end ({fit_end_min:.0f}min)")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

    night = event_data["night_date"]
    dur = event_data["duration_min"]
    ax.set_title(f"{night} | Dome OPEN | Duration: {dur:.0f} min", fontsize=13)
    ax.set_ylabel("Temperature (C)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(xlim)

    # Panel 2: Delta T
    ax = axes[1]
    ax.plot(t_plot, dT, color="purple", linewidth=1.5, label="$\\Delta T$")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

    if plateau_idx is not None:
        ax.axvline(t_min_arr[plateau_idx], color="green", linewidth=1.5,
                   linestyle="--", label=f"Plateau @ {t_min_arr[plateau_idx]:.0f} min")

    ax.set_ylabel("$\\Delta T$ (C)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 3: Rates — dT_in/dt, dT_out/dt, and d(ΔT)/dt
    ax = axes[2]
    dt_h = np.gradient(t_h)
    dt_h[dt_h == 0] = 1.0 / 60.0
    rate_in = pd.Series(np.gradient(Ti) / dt_h).rolling(15, center=True).mean()
    rate_out = pd.Series(np.gradient(To_1h) / dt_h).rolling(15, center=True).mean()
    rate_delta = pd.Series(np.gradient(dT) / (np.gradient(t_min_arr) / 60.0)).rolling(15, center=True).mean()

    ax.plot(t_plot, rate_in, color="tab:red", linewidth=1.2, label="$dT_{in}/dt$")
    ax.plot(t_plot, rate_out, color="tab:blue", linewidth=1.2, label="$dT_{out}/dt$")
    ax.plot(t_plot, rate_delta, color="purple", linewidth=2, label="$d(\\Delta T)/dt$")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)

    # Fixed noise band: ±1 C/h
    ax.axhspan(-1.0, 1.0, color="green", alpha=0.05, label="$\\pm$1 C/h")

    if plateau_idx is not None:
        ax.axvline(t_min_arr[plateau_idx], color="green", linewidth=1.5,
                   linestyle="--", alpha=0.6)

    ax.set_xlabel("Minutes from dome open")
    ax.set_ylabel("Rate (C/h)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    t_str = pd.Timestamp(event_data["t_start"]).strftime("%H%M")
    fname = f"transient_{night}_{t_str}_open.png"
    plt.savefig(output_dir / fname, bbox_inches="tight", dpi=120)
    plt.close()
    return fname


def process_night_events(args_tuple):
    """Process all open events for one night. Designed for multiprocessing."""
    night, night_df, min_dur, max_dur, do_plot, fig_dir, single_only, fit_window_min = args_tuple

    transitions = detect_dome_transitions(night_df)
    if not transitions:
        return []

    # Only open events
    open_events = [e for e in transitions if e["event_type"] == "open"]

    # Filter to single-opening nights only
    if single_only and len(open_events) != 1:
        return []

    results = []

    for event in open_events:
        result = process_event(night_df, event, min_dur, max_dur,
                               fit_window_min=fit_window_min)
        if result is None:
            continue

        if do_plot:
            plot_transient(result, fig_dir)

        save_data = {k: v for k, v in result.items() if not k.startswith("_")}
        results.append(save_data)

    return results


def main():
    parser = argparse.ArgumentParser(description="Dome transient event analysis (open events only)")
    parser.add_argument("--min-duration", type=int, default=10,
                        help="Minimum event duration in minutes")
    parser.add_argument("--max-duration", type=int, default=None,
                        help="Maximum event duration in minutes")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--single-only", action="store_true",
                        help="Only process nights with a single dome opening")
    parser.add_argument("--fit-window", type=int, default=None,
                        help="Fixed fit window in minutes (overrides plateau)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    args = parser.parse_args()

    from multiprocessing import Pool, cpu_count
    n_workers = args.workers or min(cpu_count(), 16)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_thermal_data()
    nights = df["night_date"].unique()
    print(f"Found {len(nights)} nights")

    do_plot = not args.no_plot

    # Build work items
    work = []
    for night in nights:
        night_df = df[df["night_date"] == night]
        work.append((night, night_df, args.min_duration, args.max_duration,
                     do_plot, FIGURES_DIR, args.single_only, args.fit_window))

    print(f"Processing {len(work)} nights with {n_workers} workers (open events only)...")

    with Pool(n_workers) as pool:
        all_results = pool.map(process_night_events, work)

    # Flatten
    all_events = [e for night_results in all_results for e in night_results]

    if not all_events:
        print("No open transient events found")
        return

    events_df = pd.DataFrame(all_events)
    events_df.to_csv(RESULTS_PATH, index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"OPEN TRANSIENT EVENT CATALOG")
    print(f"{'='*70}")
    print(f"Total open events: {len(events_df)}")

    for model, name in [("m2", "Forced Ventilation"), ("m2b", "Forced + Offset"),
                         ("m6", "Stratified")]:
        col = f"{model}_tau"
        if col in events_df.columns:
            tau = events_df[col].dropna() * 60
            rmse = events_df[f"{model}_rmse"].dropna()
            print(f"\n  {name}:")
            print(f"    tau: median={tau.median():.0f}min, mean={tau.mean():.0f}min, "
                  f"IQR=[{tau.quantile(0.25):.0f}, {tau.quantile(0.75):.0f}]min")
            print(f"    RMSE: median={rmse.median():.3f} C")
            if f"{model}_alpha" in events_df.columns:
                alpha = events_df[f"{model}_alpha"].dropna()
                print(f"    alpha: median={alpha.median():.2f}")
            if f"{model}_dT_eq" in events_df.columns:
                dTeq = events_df[f"{model}_dT_eq"].dropna()
                if len(dTeq) > 0:
                    print(f"    dT_eq: median={dTeq.median():.2f} C")
            if f"{model}_t0" in events_df.columns:
                t0 = events_df[f"{model}_t0"].dropna()
                if len(t0) > 0:
                    print(f"    t0: median={t0.median():.0f}min")

    # Plateau stats
    has_plateau = events_df["plateau_min"].notna().sum()
    plat = events_df["plateau_min"].dropna()
    print(f"\n  Plateau detected: {has_plateau}/{len(events_df)} ({100*has_plateau/len(events_df):.0f}%)")
    if len(plat) > 0:
        print(f"  Plateau time: median={plat.median():.0f}min, mean={plat.mean():.0f}min")

    # Correlations
    both = events_df[["m6_tau", "wind_speed_mean"]].dropna()
    if len(both) > 5:
        corr = both["m6_tau"].corr(both["wind_speed_mean"])
        print(f"\n  M6 tau vs wind: r={corr:.2f}")

    # tau vs initial delta_T
    if "m2_tau" in events_df.columns:
        both = events_df[["m2_tau", "delta_T_at_onset"]].dropna()
        if len(both) > 5:
            corr = both["m2_tau"].corr(both["delta_T_at_onset"].abs())
            print(f"\n  tau vs |ΔT₀| correlation: r={corr:.2f}")

    # tau vs wind
    if "m2_tau" in events_df.columns and "wind_speed_mean" in events_df.columns:
        both = events_df[["m2_tau", "wind_speed_mean"]].dropna()
        if len(both) > 5:
            corr = both["m2_tau"].corr(both["wind_speed_mean"])
            print(f"  tau vs wind correlation: r={corr:.2f}")

    print(f"\nCatalog saved to: {RESULTS_PATH}")
    print(f"Plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
