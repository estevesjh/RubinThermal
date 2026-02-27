#!/usr/bin/env python3
"""Quick test: compare ESS 113 vs aboveMirrorTemp as ground coupling on open transients."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from dome_thermal import smooth_temperatures, simulate_ode

DATA_PATH = "/sdf/data/rubin/user/esteves/thermal_analysis/thermal_data_all_observing_nights.csv"

def fit_stratified(t_h, T_in, T_out, T_ground):
    T_in0 = T_in[0]
    T_gnd = pd.Series(T_ground).interpolate(limit_direction="both").values

    def res(p):
        tau, alpha, k = p
        pred = simulate_ode(t_h, T_out, T_in0,
            {"tau": tau, "alpha": alpha, "k": k, "T_ground_data": T_gnd},
            "ground_coupled")
        return pred - T_in

    r = least_squares(res, x0=[0.5, 0.5, 0.2], bounds=([0.01, 0.01, -2], [12, 0.99, 2]))
    tau, alpha, k = r.x
    pred = simulate_ode(t_h, T_out, T_in0,
        {"tau": tau, "alpha": alpha, "k": k, "T_ground_data": T_gnd},
        "ground_coupled")
    rmse = np.sqrt(np.mean((pred - T_in) ** 2))
    return tau, alpha, k, rmse


df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True, low_memory=False)
cols = ["inside_m2_temp_mean", "outside_temp_mean", "inside_m1m3_temp_mean",
        "aboveMirrorTemperature_mean", "dome_open", "night_date"]
df = df[[c for c in cols if c in df.columns]]
print(f"Loaded {len(df)} rows, {df['night_date'].nunique()} nights")

results_m1m3 = []
results_above = []

for night in df["night_date"].unique():
    ndf = df[df["night_date"] == night]
    dome = ndf["dome_open"].fillna(False)
    try:
        dome = dome.astype(bool)
    except (ValueError, TypeError):
        dome = dome.map(lambda x: str(x).lower() == "true")

    trans = dome.astype(int).diff().fillna(0)
    openings = ndf.index[trans == 1].tolist()
    closings = ndf.index[trans == -1].tolist()

    for t_open in openings:
        next_close = [c for c in closings if c > t_open]
        t_end = next_close[0] if next_close else ndf.index[-1]
        dur = (t_end - t_open).total_seconds() / 60
        if dur < 15 or dur > 360:
            continue

        pre = t_open - pd.Timedelta(minutes=5)
        mask = (ndf.index >= pre) & (ndf.index <= t_end)
        seg = ndf.loc[mask]
        if len(seg) < 15:
            continue

        T_in_raw = seg["inside_m2_temp_mean"].values
        T_out_raw = seg["outside_temp_mean"].values
        if np.isfinite(T_in_raw).sum() < 10 or np.isfinite(T_out_raw).sum() < 10:
            continue

        T_in = smooth_temperatures(T_in_raw, window_min=5, polyorder=2)
        T_out = smooth_temperatures(T_out_raw, window_min=5, polyorder=2)
        t_h = (seg.index - t_open).total_seconds() / 3600.0
        valid = np.isfinite(T_in) & np.isfinite(T_out)

        # ESS 113
        if "inside_m1m3_temp_mean" in seg.columns:
            T_raw = seg["inside_m1m3_temp_mean"].values
            if np.isfinite(T_raw).sum() > 10:
                T_sm = smooth_temperatures(T_raw, window_min=5, polyorder=2)
                v = valid & np.isfinite(T_sm)
                if v.sum() > 10:
                    try:
                        tau, alpha, k, rmse = fit_stratified(t_h[v], T_in[v], T_out[v], T_sm[v])
                        results_m1m3.append({"night": night, "tau": tau * 60, "alpha": alpha, "k": k, "rmse": rmse})
                    except Exception:
                        pass

        # aboveMirrorTemperature
        if "aboveMirrorTemperature_mean" in seg.columns:
            T_raw = seg["aboveMirrorTemperature_mean"].values
            if np.isfinite(T_raw).sum() > 10:
                T_sm = smooth_temperatures(T_raw, window_min=5, polyorder=2)
                v = valid & np.isfinite(T_sm)
                if v.sum() > 10:
                    try:
                        tau, alpha, k, rmse = fit_stratified(t_h[v], T_in[v], T_out[v], T_sm[v])
                        results_above.append({"night": night, "tau": tau * 60, "alpha": alpha, "k": k, "rmse": rmse})
                    except Exception:
                        pass

df_m1m3 = pd.DataFrame(results_m1m3)
df_above = pd.DataFrame(results_above)

print(f"\n{'='*70}")
print(f"OPEN EVENTS: Stratified Exchange with ESS 113 (inside_m1m3)")
print(f"{'='*70}")
print(f"Events: {len(df_m1m3)}")
if len(df_m1m3) > 0:
    t = df_m1m3["tau"]
    print(f"tau (min): median={t.median():.0f}, mean={t.mean():.0f}, IQR=[{t.quantile(0.25):.0f}, {t.quantile(0.75):.0f}]")
    print(f"alpha:     median={df_m1m3['alpha'].median():.2f}")
    print(f"RMSE:      median={df_m1m3['rmse'].median():.3f}")

print(f"\n{'='*70}")
print(f"OPEN EVENTS: Stratified Exchange with aboveMirrorTemperature")
print(f"{'='*70}")
print(f"Events: {len(df_above)}")
if len(df_above) > 0:
    t = df_above["tau"]
    print(f"tau (min): median={t.median():.0f}, mean={t.mean():.0f}, IQR=[{t.quantile(0.25):.0f}, {t.quantile(0.75):.0f}]")
    print(f"alpha:     median={df_above['alpha'].median():.2f}")
    print(f"RMSE:      median={df_above['rmse'].median():.3f}")

# Head-to-head comparison
if len(df_m1m3) > 0 and len(df_above) > 0:
    # Group by night to handle multiple events per night
    m1 = df_m1m3.groupby("night").agg({"rmse": "mean", "tau": "median", "alpha": "median"}).rename(
        columns={"rmse": "rmse_m1m3", "tau": "tau_m1m3", "alpha": "alpha_m1m3"})
    m2 = df_above.groupby("night").agg({"rmse": "mean", "tau": "median", "alpha": "median"}).rename(
        columns={"rmse": "rmse_above", "tau": "tau_above", "alpha": "alpha_above"})
    merged = m1.join(m2, how="inner")

    if len(merged) > 5:
        wins_m1m3 = (merged["rmse_m1m3"] < merged["rmse_above"]).sum()
        print(f"\n{'='*70}")
        print(f"HEAD-TO-HEAD on {len(merged)} nights with both sensors")
        print(f"{'='*70}")
        print(f"ESS 113 wins:        {wins_m1m3} ({100*wins_m1m3/len(merged):.0f}%)")
        print(f"aboveMirror wins:    {len(merged)-wins_m1m3} ({100*(len(merged)-wins_m1m3)/len(merged):.0f}%)")
        print(f"Mean RMSE ESS 113:   {merged['rmse_m1m3'].mean():.4f}")
        print(f"Mean RMSE aboveMir:  {merged['rmse_above'].mean():.4f}")
        print(f"Median tau ESS 113:  {merged['tau_m1m3'].median():.0f} min")
        print(f"Median tau aboveMir: {merged['tau_above'].median():.0f} min")
        print(f"Median α ESS 113:    {merged['alpha_m1m3'].median():.2f}")
        print(f"Median α aboveMir:   {merged['alpha_above'].median():.2f}")
