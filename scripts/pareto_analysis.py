#!/usr/bin/env python3
"""
Pareto frontier analysis for thermal control tradeoffs.

Analyzes the tradeoff between tracking error and gradient penalty across
different max_rate and tau_gradient values.

The two competing objectives are:
1. Minimize tracking error: Keep mirror at ambient - 0.3C
2. Minimize gradient penalty: Avoid internal temperature gradients

The gradient penalty models thermal stress from differential expansion
between surface and bulk temperatures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

from rubin_thermal import (
    CONFIG,
    load_temperature_data,
    build_day_database,
    train_test_split,
    interpolate_temp,
    compute_rate,
    make_linear_kernel,
    ThermalModel,
)
from rubin_thermal.config import get_figures_path, get_project_root


# Parameter space to sweep
MAX_RATES = [0.3, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0]  # C/hour
TAU_GRADIENTS = [2.0, 2.5, 3.0]  # hours
PENALTY_WEIGHTS = [0.5, 1.0, 2.0]  # weighting for gradient term

# Current operating point for reference
CURRENT_MAX_RATE = 0.7


def run_simulation(day, max_rate, tau_gradient, tau_thermal=3.0, cold_bias=0.3, dt=0.25):
    """
    Run thermal simulation with two-zone model.

    The two-zone model tracks:
    - T_surface: responds to setpoint changes (controlled temperature)
    - T_bulk: lags behind surface, represents interior thermal mass

    Parameters
    ----------
    day : dict
        Day data dictionary with hours_from_sunset, temps, T_sunset
    max_rate : float
        Maximum setpoint change rate (C/hour)
    tau_gradient : float
        Time constant for surface-to-bulk heat transfer (hours)
    tau_thermal : float
        Mirror thermal time constant for HVAC response (hours)
    cold_bias : float
        Target temperature below ambient (C)
    dt : float
        Simulation timestep (hours)

    Returns
    -------
    dict
        Simulation results with tracking and gradient metrics
    """
    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]

    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    t_sim = hours.copy()
    n = len(t_sim)

    # State arrays
    T_setpoint = np.zeros(n)
    T_surface = np.zeros(n)  # Surface temperature (responds to setpoint)
    T_bulk = np.zeros(n)     # Bulk temperature (lags behind surface)
    T_ambient = np.zeros(n)
    phase = np.zeros(n, dtype=int)

    # Initial conditions: mirror at target temperature
    T_surface[0] = T_sunset - cold_bias
    T_bulk[0] = T_sunset - cold_bias

    # Maximum setpoint change per timestep
    max_change = max_rate * dt

    for i in range(n):
        t = t_sim[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb

        # Compute rate using centered difference
        rate = compute_rate(hours, temps, t, 1.0)

        # Three-phase setpoint calculation
        if t < T1:
            # Phase 1: Fixed at sunset prediction
            T_setpoint[i] = T_sunset - cold_bias
            phase[i] = 1
        elif t < T2:
            # Phase 2: Ramp from fixed to tracking
            alpha = (t - T1) / (T2 - T1)
            fixed = T_sunset - cold_bias
            tracking = T_amb - cold_bias + 0.5 * tau_thermal * rate
            T_setpoint[i] = (1 - alpha) * fixed + alpha * tracking
            phase[i] = 2
        else:
            # Phase 3: Weighted lookahead
            weights, times = make_linear_kernel()
            weighted_temp = sum(
                w * interpolate_temp(hours, temps, t + dt_look)
                for w, dt_look in zip(weights, times)
            )
            T_setpoint[i] = weighted_temp + 0.5 * tau_thermal * rate - cold_bias
            phase[i] = 3

        # Apply rate limiting
        if i > 0:
            delta = T_setpoint[i] - T_setpoint[i - 1]
            if abs(delta) > max_change:
                T_setpoint[i] = T_setpoint[i - 1] + np.sign(delta) * max_change

            # Surface responds to setpoint (HVAC control)
            # First-order response: dT/dt = (T_setpoint - T_surface) / tau_thermal
            T_surface[i] = T_surface[i - 1] + (T_setpoint[i] - T_surface[i - 1]) / tau_thermal * dt

            # Bulk responds to surface (internal heat diffusion)
            # First-order response: dT/dt = (T_surface - T_bulk) / tau_gradient
            T_bulk[i] = T_bulk[i - 1] + (T_surface[i - 1] - T_bulk[i - 1]) / tau_gradient * dt
        else:
            T_surface[0] = T_sunset - cold_bias
            T_bulk[0] = T_sunset - cold_bias

    # Compute tracking metrics (overnight only, t >= 0)
    overnight_mask = t_sim >= 0
    T_target = T_ambient - cold_bias
    tracking_errors = T_surface - T_target

    overnight_errors = tracking_errors[overnight_mask]
    tracking_rms = np.sqrt(np.mean(overnight_errors ** 2))
    pct_within_tolerance = 100 * np.mean(np.abs(overnight_errors) <= 0.3)

    # Compute gradient metrics (overnight only)
    gradient = T_surface - T_bulk  # Surface - Bulk temperature difference
    overnight_gradient = gradient[overnight_mask]

    peak_gradient = np.max(np.abs(overnight_gradient))
    mean_gradient = np.mean(np.abs(overnight_gradient))
    # Gradient integral: accumulated squared gradient over time (damage metric)
    gradient_integral = np.sum(overnight_gradient ** 2) * dt

    # Sunset error
    sunset_idx = np.argmin(np.abs(t_sim))
    error_at_sunset = tracking_errors[sunset_idx]

    return {
        # Tracking metrics
        "tracking_rms": tracking_rms,
        "pct_within_tolerance": pct_within_tolerance,
        "error_at_sunset": error_at_sunset,
        # Gradient metrics
        "peak_gradient": peak_gradient,
        "mean_gradient": mean_gradient,
        "gradient_integral": gradient_integral,
        # Time series for debugging
        "t": t_sim,
        "T_surface": T_surface,
        "T_bulk": T_bulk,
        "T_setpoint": T_setpoint,
        "T_ambient": T_ambient,
        "gradient": gradient,
    }


def evaluate_configuration(days, max_rate, tau_gradient):
    """
    Evaluate a configuration across all nights.

    Parameters
    ----------
    days : list
        List of day dictionaries
    max_rate : float
        Maximum setpoint rate (C/hour)
    tau_gradient : float
        Gradient time constant (hours)

    Returns
    -------
    dict
        Aggregated metrics across all nights
    """
    all_tracking_rms = []
    all_pct_within = []
    all_peak_gradient = []
    all_mean_gradient = []
    all_gradient_integral = []
    all_sunset_errors = []

    for day in days:
        result = run_simulation(day, max_rate, tau_gradient)
        all_tracking_rms.append(result["tracking_rms"])
        all_pct_within.append(result["pct_within_tolerance"])
        all_peak_gradient.append(result["peak_gradient"])
        all_mean_gradient.append(result["mean_gradient"])
        all_gradient_integral.append(result["gradient_integral"])
        all_sunset_errors.append(result["error_at_sunset"])

    return {
        "max_rate": max_rate,
        "tau_gradient": tau_gradient,
        "tracking_rms": np.mean(all_tracking_rms),
        "tracking_rms_std": np.std(all_tracking_rms),
        "pct_within_tolerance": np.mean(all_pct_within),
        "peak_gradient": np.mean(all_peak_gradient),
        "peak_gradient_max": np.max(all_peak_gradient),
        "mean_gradient": np.mean(all_mean_gradient),
        "gradient_integral": np.mean(all_gradient_integral),
        "gradient_integral_std": np.std(all_gradient_integral),
        "sunset_error_mean": np.mean(all_sunset_errors),
        "sunset_error_std": np.std(all_sunset_errors),
    }


def identify_pareto_frontier(results_df, obj1_col, obj2_col, minimize_both=True):
    """
    Identify Pareto-optimal points.

    A point is Pareto-optimal if no other point is better in both objectives.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    obj1_col : str
        Column name for first objective
    obj2_col : str
        Column name for second objective
    minimize_both : bool
        If True, both objectives are minimized

    Returns
    -------
    np.ndarray
        Boolean mask for Pareto-optimal points
    """
    n = len(results_df)
    pareto_mask = np.ones(n, dtype=bool)

    obj1 = results_df[obj1_col].values
    obj2 = results_df[obj2_col].values

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            if minimize_both:
                # j dominates i if j is <= in both and < in at least one
                if obj1[j] <= obj1[i] and obj2[j] <= obj2[i]:
                    if obj1[j] < obj1[i] or obj2[j] < obj2[i]:
                        pareto_mask[i] = False
                        break
            else:
                raise NotImplementedError("Only minimize_both=True supported")

    return pareto_mask


def compute_combined_objective(tracking_rms, gradient_integral, penalty_weight):
    """
    Compute combined objective function.

    The objective balances tracking performance against gradient-induced stress.
    The gradient term is normalized so that penalty_weight=1.0 represents
    equal weighting between a 0.1C tracking improvement and a 1 C^2*h gradient
    reduction.

    Parameters
    ----------
    tracking_rms : float
        Tracking RMS error (C)
    gradient_integral : float
        Gradient integral (C^2 * hours)
    penalty_weight : float
        Weight for gradient penalty relative to tracking

    Returns
    -------
    float
        Combined objective value
    """
    # Scale gradient to same order as tracking_rms
    # Use gradient_integral directly with scaling factor
    # At baseline (rate=0.7, tau_grad=3.0): tracking ~0.44, gradient ~3.5
    # We want penalty_weight=1.0 to make these comparable
    normalized_gradient = gradient_integral * 0.15  # Scale to ~0.5 range
    return tracking_rms + penalty_weight * normalized_gradient


def main():
    """Run Pareto frontier analysis."""
    print("=" * 60)
    print("PARETO FRONTIER ANALYSIS")
    print("Tracking Error vs Gradient Penalty Tradeoff")
    print("=" * 60)

    # Set up paths
    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)
    results_path = get_project_root() / "results"
    results_path.mkdir(exist_ok=True)

    # Load data
    print("\nLoading temperature data...")
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Using {len(test_days)} test nights (odd dates)")

    # Run parameter sweep
    print("\nRunning parameter sweep...")
    print(f"  max_rate: {MAX_RATES}")
    print(f"  tau_gradient: {TAU_GRADIENTS}")

    all_results = []
    total_configs = len(MAX_RATES) * len(TAU_GRADIENTS)
    config_num = 0

    for tau_grad in TAU_GRADIENTS:
        for max_r in MAX_RATES:
            config_num += 1
            print(f"  [{config_num}/{total_configs}] max_rate={max_r:.2f}, tau_gradient={tau_grad:.1f}...", end="", flush=True)
            result = evaluate_configuration(test_days, max_r, tau_grad)
            all_results.append(result)
            print(f" RMS={result['tracking_rms']:.3f}, Gradient={result['gradient_integral']:.2f}")

    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_df.to_csv(results_path / "pareto_results.csv", index=False)
    print(f"\nSaved results to: {results_path / 'pareto_results.csv'}")

    # =========================================================================
    # Figure 1: Pareto Frontier (3 panels, one per tau_gradient)
    # =========================================================================
    print("\nGenerating Figure 1: Pareto Frontier...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Colormap for max_rate
    cmap = plt.cm.viridis
    rate_norm = plt.Normalize(min(MAX_RATES), max(MAX_RATES))

    for idx, tau_grad in enumerate(TAU_GRADIENTS):
        ax = axes[idx]
        subset = results_df[results_df["tau_gradient"] == tau_grad].copy()

        # Identify Pareto frontier
        pareto_mask = identify_pareto_frontier(subset, "tracking_rms", "gradient_integral")
        subset["is_pareto"] = pareto_mask

        # Sort Pareto points for line plotting
        pareto_points = subset[pareto_mask].sort_values("gradient_integral")

        # Plot all points
        scatter = ax.scatter(
            subset["gradient_integral"],
            subset["tracking_rms"],
            c=subset["max_rate"],
            cmap=cmap,
            norm=rate_norm,
            s=100,
            alpha=0.7,
            edgecolors="gray",
            linewidth=0.5,
            zorder=2,
        )

        # Highlight Pareto frontier
        if len(pareto_points) > 1:
            ax.plot(
                pareto_points["gradient_integral"],
                pareto_points["tracking_rms"],
                "k-",
                linewidth=2,
                alpha=0.8,
                zorder=1,
                label="Pareto Frontier",
            )

        # Mark Pareto-optimal points
        ax.scatter(
            subset[pareto_mask]["gradient_integral"],
            subset[pareto_mask]["tracking_rms"],
            c=subset[pareto_mask]["max_rate"],
            cmap=cmap,
            norm=rate_norm,
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=1.5,
            zorder=3,
        )

        # Mark current operating point (max_rate=0.7)
        current = subset[np.isclose(subset["max_rate"], CURRENT_MAX_RATE)]
        if len(current) > 0:
            ax.scatter(
                current["gradient_integral"].values[0],
                current["tracking_rms"].values[0],
                c="red",
                s=250,
                marker="o",
                edgecolors="black",
                linewidth=2,
                zorder=4,
                label=f"Current (rate={CURRENT_MAX_RATE})",
            )

        ax.set_xlabel("Gradient Integral (C^2 * hours)", fontsize=11)
        ax.set_ylabel("Tracking RMS (C)", fontsize=11)
        ax.set_title(f"tau_gradient = {tau_grad:.1f}h", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Max Rate (C/hour)", fontsize=11)

    plt.suptitle(
        "Pareto Frontier: Tracking Error vs Gradient Penalty\n"
        "Stars = Pareto-optimal, Red circle = Current operating point",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_path / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {figures_path / 'pareto_frontier.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: Optimal Rate vs Tau for Different Penalty Weights
    # =========================================================================
    print("\nGenerating Figure 2: Optimal Rate vs Tau...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each penalty weight, find optimal max_rate for each tau_gradient
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    markers = ["o", "s", "^"]

    optimal_configs = []

    for pw_idx, penalty_weight in enumerate(PENALTY_WEIGHTS):
        optimal_rates = []
        optimal_values = []

        for tau_grad in TAU_GRADIENTS:
            subset = results_df[results_df["tau_gradient"] == tau_grad].copy()

            # Compute combined objective
            subset["combined"] = subset.apply(
                lambda row: compute_combined_objective(
                    row["tracking_rms"], row["gradient_integral"], penalty_weight
                ),
                axis=1,
            )

            # Find optimal
            best_idx = subset["combined"].idxmin()
            best_row = subset.loc[best_idx]
            optimal_rates.append(best_row["max_rate"])
            optimal_values.append(best_row["combined"])

            optimal_configs.append({
                "penalty_weight": penalty_weight,
                "tau_gradient": tau_grad,
                "optimal_max_rate": best_row["max_rate"],
                "tracking_rms": best_row["tracking_rms"],
                "gradient_integral": best_row["gradient_integral"],
                "combined_objective": best_row["combined"],
            })

        ax.plot(
            TAU_GRADIENTS,
            optimal_rates,
            f"{markers[pw_idx]}-",
            color=colors[pw_idx],
            markersize=12,
            linewidth=2.5,
            label=f"Penalty weight = {penalty_weight}",
        )

    # Add reference line for current operating point
    ax.axhline(
        y=CURRENT_MAX_RATE,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Current rate = {CURRENT_MAX_RATE}",
    )

    ax.set_xlabel("Gradient Time Constant (tau_gradient, hours)", fontsize=12)
    ax.set_ylabel("Optimal Max Rate (C/hour)", fontsize=12)
    ax.set_title(
        "Optimal Rate Selection vs Gradient Physics Assumption\n"
        "Higher penalty weight favors lower rates (less gradient stress)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(TAU_GRADIENTS)
    ax.set_ylim(0, max(MAX_RATES) + 0.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_path / "optimal_rate_vs_tau.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {figures_path / 'optimal_rate_vs_tau.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Tradeoff Surface (3D or contour)
    # =========================================================================
    print("\nGenerating Figure 3: Tradeoff Surface...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, penalty_weight in enumerate(PENALTY_WEIGHTS):
        ax = axes[idx]

        # Create grid for contour plot
        rate_vals = np.array(MAX_RATES)
        tau_vals = np.array(TAU_GRADIENTS)

        # Build 2D grid of combined objective values
        Z = np.zeros((len(TAU_GRADIENTS), len(MAX_RATES)))

        for i, tau_grad in enumerate(TAU_GRADIENTS):
            for j, max_r in enumerate(MAX_RATES):
                row = results_df[
                    (np.isclose(results_df["tau_gradient"], tau_grad)) &
                    (np.isclose(results_df["max_rate"], max_r))
                ].iloc[0]
                Z[i, j] = compute_combined_objective(
                    row["tracking_rms"], row["gradient_integral"], penalty_weight
                )

        # Create contour plot
        X, Y = np.meshgrid(rate_vals, tau_vals)
        contour = ax.contourf(X, Y, Z, levels=15, cmap="RdYlGn_r")
        ax.contour(X, Y, Z, levels=15, colors="black", linewidths=0.5, alpha=0.3)

        # Mark minimum
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax.scatter(
            rate_vals[min_idx[1]],
            tau_vals[min_idx[0]],
            c="white",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            zorder=5,
            label=f"Optimal: rate={rate_vals[min_idx[1]]:.2f}",
        )

        # Mark current operating point
        current_idx = np.where(np.isclose(rate_vals, CURRENT_MAX_RATE))[0]
        if len(current_idx) > 0:
            for tau_i, tau in enumerate(tau_vals):
                ax.scatter(
                    CURRENT_MAX_RATE,
                    tau,
                    c="red",
                    s=100,
                    marker="o",
                    edgecolors="black",
                    linewidth=1.5,
                    zorder=4,
                )

        ax.set_xlabel("Max Rate (C/hour)", fontsize=11)
        ax.set_ylabel("tau_gradient (hours)", fontsize=11)
        ax.set_title(f"Penalty Weight = {penalty_weight}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Combined Objective", fontsize=10)

    plt.suptitle(
        "Optimization Landscape: Combined Objective\n"
        "(Lower = Better) White star = Optimal, Red circle = Current",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_path / "tradeoff_surface.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {figures_path / 'tradeoff_surface.png'}")
    plt.close()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: OPTIMAL CONFIGURATIONS BY SCENARIO")
    print("=" * 90)

    # Find improvement vs current for each scenario
    current_results = {}
    for tau_grad in TAU_GRADIENTS:
        current_row = results_df[
            (np.isclose(results_df["tau_gradient"], tau_grad)) &
            (np.isclose(results_df["max_rate"], CURRENT_MAX_RATE))
        ].iloc[0]
        current_results[tau_grad] = current_row

    print(f"\n{'tau_grad':<10} {'Penalty':<10} {'Opt Rate':<10} {'Track RMS':<12} {'Gradient':<12} {'vs 0.7 Rate':<15}")
    print("-" * 90)

    for config in optimal_configs:
        tau_grad = config["tau_gradient"]
        penalty = config["penalty_weight"]
        opt_rate = config["optimal_max_rate"]
        track_rms = config["tracking_rms"]
        grad_int = config["gradient_integral"]

        # Compute improvement vs current
        current = current_results[tau_grad]
        current_combined = compute_combined_objective(
            current["tracking_rms"], current["gradient_integral"], penalty
        )
        opt_combined = config["combined_objective"]
        improvement = 100 * (current_combined - opt_combined) / current_combined

        improvement_str = f"{improvement:+.1f}%" if abs(improvement) > 0.1 else "same"

        print(f"{tau_grad:<10.1f} {penalty:<10.1f} {opt_rate:<10.2f} {track_rms:<12.3f} {grad_int:<12.2f} {improvement_str:<15}")

    # Detailed results for current operating point
    print("\n" + "=" * 90)
    print("CURRENT OPERATING POINT (max_rate = 0.7 C/hour)")
    print("=" * 90)
    print(f"\n{'tau_gradient':<15} {'Tracking RMS':<15} {'% within 0.3C':<15} {'Peak Gradient':<15} {'Gradient Int':<15}")
    print("-" * 90)

    for tau_grad in TAU_GRADIENTS:
        row = current_results[tau_grad]
        print(f"{tau_grad:<15.1f} {row['tracking_rms']:<15.3f} {row['pct_within_tolerance']:<15.1f} {row['peak_gradient']:<15.3f} {row['gradient_integral']:<15.2f}")

    # Key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    print("""
1. TRACKING vs GRADIENT TRADEOFF:
   - Higher max_rate improves tracking but increases gradient penalty
   - Lower max_rate reduces gradients but degrades tracking performance

2. TAU_GRADIENT IMPACT:
   - Larger tau_gradient (slower internal diffusion) increases gradient penalties
   - Smaller tau_gradient allows surface/bulk to equilibrate faster

3. RECOMMENDATIONS BY SCENARIO:
   - If gradient stress is minor concern (weight=0.5): Use higher rate (~1.0-1.5 C/h)
   - If gradient stress is moderate (weight=1.0): Current rate (~0.7 C/h) is reasonable
   - If gradient stress is major concern (weight=2.0): Consider lower rate (~0.5 C/h)

4. CURRENT OPERATING POINT (0.7 C/h):
   - Represents a balanced choice on the Pareto frontier
   - Near-optimal for moderate gradient penalty assumptions
""")

    print(f"\nResults saved to: {results_path / 'pareto_results.csv'}")
    print("Figures saved to:")
    print(f"  - {figures_path / 'pareto_frontier.png'}")
    print(f"  - {figures_path / 'optimal_rate_vs_tau.png'}")
    print(f"  - {figures_path / 'tradeoff_surface.png'}")


if __name__ == "__main__":
    main()
