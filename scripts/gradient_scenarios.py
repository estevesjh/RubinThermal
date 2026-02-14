#!/usr/bin/env python3
"""
Gradient penalty scenarios for thermal control optimization.

This script defines and evaluates three scenarios for balancing tracking
performance against internal gradient penalties in the M1M3 mirror:

Scenario 1: Conservative (Honeycomb-Safe)
    - Assumes worst-case honeycomb thermal patterns
    - Very restrictive gradient penalty
    - Prioritizes mirror safety over tracking accuracy

Scenario 2: Moderate (Balanced)
    - Balanced approach between tracking and gradients
    - Represents typical operating conditions

Scenario 3: Aggressive (Performance-Focused)
    - Prioritizes tracking performance
    - Accepts higher gradients during rapid ambient changes
    - Assumes faster gradient relaxation

The script sweeps over max_rate values to find the optimal rate limit
for each scenario, considering both tracking error and gradient damage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rubin_thermal import (
    CONFIG,
    build_day_database,
    compute_rate,
    interpolate_temp,
    load_temperature_data,
    make_linear_kernel,
    train_test_split,
)
from rubin_thermal.config import get_figures_path
from rubin_thermal.gradient_model import TwoZoneThermalModel


@dataclass
class GradientScenario:
    """
    Configuration for a gradient penalty scenario.

    Parameters
    ----------
    name : str
        Human-readable scenario name
    description : str
        Brief description of the scenario
    tau_gradient : float
        Gradient relaxation time constant (hours)
    gradient_threshold : float
        Gradient threshold for penalty activation (C)
    penalty_weight : float
        Weight applied to gradient penalty in combined objective
    color : str
        Color for plotting
    """

    name: str
    description: str
    tau_gradient: float
    gradient_threshold: float
    penalty_weight: float
    color: str


# Define the three scenarios (with more differentiated weights)
SCENARIOS = {
    "conservative": GradientScenario(
        name="Conservative (Honeycomb-Safe)",
        description="Assumes worst-case honeycomb thermal patterns",
        tau_gradient=3.0,
        gradient_threshold=0.3,
        penalty_weight=5.0,  # High weight - strongly penalize gradients
        color="tab:blue",
    ),
    "moderate": GradientScenario(
        name="Moderate (Balanced)",
        description="Balanced approach between tracking and gradients",
        tau_gradient=3.0,
        gradient_threshold=0.5,
        penalty_weight=1.0,
        color="tab:green",
    ),
    "aggressive": GradientScenario(
        name="Aggressive (Performance-Focused)",
        description="Prioritizes tracking performance",
        tau_gradient=3.0,
        gradient_threshold=0.8,
        penalty_weight=0.2,  # Low weight - prioritize tracking
        color="tab:orange",
    ),
}

# Max rate values to sweep (finer resolution)
MAX_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0]


def compute_gradient_penalty(
    gradient: float,
    rate_of_change: float,
    scenario: GradientScenario,
) -> Tuple[float, float]:
    """
    Compute image quality penalty from thermal gradients.

    The penalty has two components:
    1. Instantaneous gradient penalty: |gradient|^2 above threshold
    2. Rate penalty: |dT/dt|^2 weighted by sensitivity

    Parameters
    ----------
    gradient : float
        Current internal gradient (T_surface - T_bulk) in C
    rate_of_change : float
        Current setpoint rate of change in C/h
    scenario : GradientScenario
        Scenario parameters

    Returns
    -------
    tuple : (instant_penalty, rate_penalty)
        instant_penalty : float
            Penalty from gradient magnitude
        rate_penalty : float
            Penalty from rate of change
    """
    threshold = scenario.gradient_threshold

    # Gradient magnitude penalty (quadratic above threshold)
    if abs(gradient) > threshold:
        excess = abs(gradient) - threshold
        instant_penalty = excess**2
    else:
        instant_penalty = 0.0

    # Rate penalty (always quadratic, scaled by tau_gradient)
    # Faster rate changes create larger transient gradients
    rate_sensitivity = scenario.tau_gradient / 3.0  # Normalize to typical tau
    rate_penalty = (rate_of_change * rate_sensitivity) ** 2 * 0.1

    return instant_penalty, rate_penalty


def simulate_scenario(
    day: dict,
    scenario: GradientScenario,
    max_rate: float,
) -> dict:
    """
    Simulate one night with a given scenario and max_rate.

    Parameters
    ----------
    day : dict
        Day data from build_day_database
    scenario : GradientScenario
        Scenario parameters
    max_rate : float
        Maximum setpoint rate of change (C/h)

    Returns
    -------
    dict
        Simulation results with tracking and gradient metrics
    """
    # Control parameters from config
    T1 = CONFIG["control"]["t1"]
    T2 = CONFIG["control"]["t2"]
    cold_bias = CONFIG["physics"]["cold_bias"]
    tau = CONFIG["physics"]["tau"]
    dt = CONFIG["physics"]["dt"]

    hours = day["hours_from_sunset"]
    temps = day["temps"]
    T_sunset = day["T_sunset"]

    # Create two-zone model with scenario parameters
    # Fast surface response (0.25h) with active cooling
    # Rate-gradient coupling makes gradient proportional to rate of change
    model = TwoZoneThermalModel(
        tau_bulk=tau,
        tau_surface=1.0,  # Fast with active cooling
        tau_gradient=scenario.tau_gradient,
        max_rate=max_rate,
        dt=dt,
        cold_bias=cold_bias,
        rate_gradient_coupling=0.3,  # Gradient scales with rate
    )

    # Initialize at sunset temperature
    model.initialize(T_sunset - cold_bias)

    n = len(hours)
    T_setpoint = np.zeros(n)
    T_surface = np.zeros(n)
    T_bulk = np.zeros(n)
    gradients = np.zeros(n)
    T_ambient = np.zeros(n)
    instant_penalties = np.zeros(n)
    rate_penalties = np.zeros(n)

    # Initial conditions
    T_setpoint[0] = T_sunset - cold_bias
    T_surface[0] = model.T_surface
    T_bulk[0] = model.T_bulk
    gradients[0] = model.gradient
    T_ambient[0] = temps[0]

    for i in range(1, n):
        t = hours[i]
        T_amb = temps[i]
        T_ambient[i] = T_amb

        # Compute rate from recent data
        rate = compute_rate(hours, temps, t, 1.0)

        # Compute desired setpoint based on control phase
        if t < T1:
            # Phase 1: Fixed setpoint
            T_sp_desired = T_sunset - cold_bias
        elif t < T2:
            # Phase 2: Linear ramp from fixed to tracking
            alpha = (t - T1) / (T2 - T1)
            fixed = T_sunset - cold_bias
            tracking = T_amb - cold_bias + 0.5 * tau * rate
            T_sp_desired = (1 - alpha) * fixed + alpha * tracking
        else:
            # Phase 3: Weighted lookahead with rate compensation
            weights, times = make_linear_kernel()
            weighted_temp = sum(
                w * interpolate_temp(hours, temps, t + dt_look)
                for w, dt_look in zip(weights, times)
            )
            T_sp_desired = weighted_temp + 0.5 * tau * rate - cold_bias

        # Step the model (rate limiting applied internally)
        T_surf, T_blk, grad = model.step(T_sp_desired)

        T_setpoint[i] = model.T_setpoint_prev
        T_surface[i] = T_surf
        T_bulk[i] = T_blk
        gradients[i] = grad

        # Compute gradient penalties
        setpoint_rate = model.get_setpoint_rate(window=4)
        instant_pen, rate_pen = compute_gradient_penalty(grad, setpoint_rate, scenario)
        instant_penalties[i] = instant_pen
        rate_penalties[i] = rate_pen

    # Compute metrics (overnight only: t >= 0)
    overnight_mask = hours >= 0

    # Tracking error (using surface temperature as representative)
    T_target = T_ambient - cold_bias
    errors = T_surface - T_target
    overnight_errors = errors[overnight_mask]
    abs_errors = np.abs(overnight_errors)

    # Tracking metrics - percentiles and thresholds
    tracking_rms = np.sqrt(np.mean(overnight_errors**2))
    tracking_median = np.median(abs_errors)
    tracking_p68 = np.percentile(abs_errors, 68)
    tracking_p95 = np.percentile(abs_errors, 95)
    tracking_within_03 = 100 * np.mean(abs_errors < 0.3)
    tracking_within_05 = 100 * np.mean(abs_errors < 0.5)
    tracking_within_10 = 100 * np.mean(abs_errors < 1.0)

    # Gradient metrics (overnight) - return raw arrays for global percentile calculation
    overnight_gradients = gradients[overnight_mask]
    abs_gradients = np.abs(overnight_gradients)
    peak_gradient = np.max(abs_gradients)  # Keep for reference
    rms_gradient = np.sqrt(np.mean(overnight_gradients**2))

    # Accumulated penalties (overnight)
    overnight_instant = instant_penalties[overnight_mask]
    overnight_rate = rate_penalties[overnight_mask]
    accumulated_penalty = np.sum(overnight_instant + overnight_rate) * dt

    # Combined objective: tracking RMS + weighted penalty
    # Normalize penalty to be in comparable units to tracking RMS
    penalty_contribution = np.sqrt(accumulated_penalty) * 0.1
    combined_objective = tracking_rms + scenario.penalty_weight * penalty_contribution

    return {
        "hours": hours,
        "T_ambient": T_ambient,
        "T_target": T_target,
        "T_setpoint": T_setpoint,
        "T_surface": T_surface,
        "T_bulk": T_bulk,
        "gradients": gradients,
        "errors": errors,
        "instant_penalties": instant_penalties,
        "rate_penalties": rate_penalties,
        "date": day["date"],
        # Tracking metrics
        "tracking_rms": tracking_rms,
        "tracking_median": tracking_median,
        "tracking_p68": tracking_p68,
        "tracking_p95": tracking_p95,
        "tracking_within_03": tracking_within_03,
        "tracking_within_05": tracking_within_05,
        "tracking_within_10": tracking_within_10,
        # Gradient metrics - return raw arrays for global percentile
        "overnight_abs_errors": abs_errors,
        "overnight_abs_gradients": abs_gradients,
        "peak_gradient": peak_gradient,
        "rms_gradient": rms_gradient,
        "accumulated_penalty": accumulated_penalty,
        "combined_objective": combined_objective,
    }


def run_sweep(
    days: List[dict],
    scenario: GradientScenario,
    max_rates: List[float],
) -> dict:
    """
    Run parameter sweep over max_rate values for a scenario.

    Parameters
    ----------
    days : list of dict
        Test day data
    scenario : GradientScenario
        Scenario to evaluate
    max_rates : list of float
        Max rate values to sweep

    Returns
    -------
    dict
        Results indexed by max_rate
    """
    results = {}

    for max_rate in max_rates:
        print(f"  max_rate = {max_rate:.2f} C/h...")

        # Collect raw data from all nights for global percentiles
        all_abs_errors = []
        all_abs_gradients = []
        all_peak_gradient = []
        all_rms_gradient = []
        all_accumulated = []
        all_combined = []

        for day in days:
            result = simulate_scenario(day, scenario, max_rate)
            # Collect raw arrays for global percentile calculation
            all_abs_errors.append(result["overnight_abs_errors"])
            all_abs_gradients.append(result["overnight_abs_gradients"])
            all_peak_gradient.append(result["peak_gradient"])
            all_rms_gradient.append(result["rms_gradient"])
            all_accumulated.append(result["accumulated_penalty"])
            all_combined.append(result["combined_objective"])

        # Concatenate all data points from all nights
        global_abs_errors = np.concatenate(all_abs_errors)
        global_abs_gradients = np.concatenate(all_abs_gradients)

        # Compute global percentiles over ALL data points
        tracking_median = np.median(global_abs_errors)
        tracking_p68 = np.percentile(global_abs_errors, 68)
        tracking_p95 = np.percentile(global_abs_errors, 95)
        within_05 = 100 * np.mean(global_abs_errors < 0.5)
        within_10 = 100 * np.mean(global_abs_errors < 1.0)

        # Gradient percentiles - median, 68th, 95th
        gradient_median = np.median(global_abs_gradients)
        gradient_p68 = np.percentile(global_abs_gradients, 68)
        gradient_p95 = np.percentile(global_abs_gradients, 95)

        results[max_rate] = {
            # Tracking metrics - global percentiles over all data
            "tracking_median_mean": tracking_median,
            "tracking_p68_mean": tracking_p68,
            "tracking_p95_mean": tracking_p95,
            "within_05_mean": within_05,
            "within_10_mean": within_10,
            # Gradient metrics - global percentiles over all data
            "gradient_median_mean": gradient_median,
            "gradient_p68_mean": gradient_p68,
            "gradient_p95_mean": gradient_p95,
            "peak_gradient_mean": np.mean(all_peak_gradient),
            "peak_gradient_std": np.std(all_peak_gradient),
            "rms_gradient_mean": np.mean(all_rms_gradient),
            "rms_gradient_std": np.std(all_rms_gradient),
            "accumulated_mean": np.mean(all_accumulated),
            "accumulated_std": np.std(all_accumulated),
            "combined_mean": np.mean(all_combined),
            "combined_std": np.std(all_combined),
        }

    return results


def find_optimal_rate(results: dict) -> float:
    """
    Find the optimal max_rate that minimizes combined objective.

    Parameters
    ----------
    results : dict
        Results from run_sweep

    Returns
    -------
    float
        Optimal max_rate value
    """
    min_combined = float("inf")
    optimal_rate = None

    for rate, metrics in results.items():
        if metrics["combined_mean"] < min_combined:
            min_combined = metrics["combined_mean"]
            optimal_rate = rate

    return optimal_rate


def print_summary_table(all_results: Dict[str, dict]) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 120)
    print("GRADIENT SCENARIOS SUMMARY")
    print("=" * 120)

    for scenario_key, (scenario, results) in all_results.items():
        optimal_rate = find_optimal_rate(results)
        print(f"\n{scenario.name}")
        print(
            f"  tau_gradient={scenario.tau_gradient}h, threshold={scenario.gradient_threshold}C, weight={scenario.penalty_weight}"
        )
        print("-" * 120)
        print(
            f"{'Rate':>8} | {'Median':>8} | {'68%ile':>8} | {'95%ile':>8} | {'<0.5C':>8} | {'<1.0C':>8} | {'Grad 95%':>10} | {'Combined':>10}"
        )
        print(
            f"{'(C/h)':>8} | {'(C)':>8} | {'(C)':>8} | {'(C)':>8} | {'(%)':>8} | {'(%)':>8} | {'(C)':>10} | {'objective':>10}"
        )
        print("-" * 120)

        for rate in MAX_RATES:
            m = results[rate]
            marker = " <-- optimal" if rate == optimal_rate else ""
            print(
                f"{rate:>8.2f} | "
                f"{m['tracking_median_mean']:>8.3f} | "
                f"{m['tracking_p68_mean']:>8.3f} | "
                f"{m['tracking_p95_mean']:>8.3f} | "
                f"{m['within_05_mean']:>8.1f} | "
                f"{m['within_10_mean']:>8.1f} | "
                f"{m['gradient_p95_mean']:>10.3f} | "
                f"{m['combined_mean']:>10.3f}{marker}"
            )


def create_figure(all_results: Dict[str, dict], figures_path: Path) -> None:
    """
    Create summary figure with tracking and gradient panels.

    Layout:
    - Top row: Tracking percentiles (median, 68%, 95%) vs max_rate (3 panels)
    - Bottom row: Tracking thresholds (% <0.5C, <1.0C) and gradient 95th %ile (3 panels)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    scenario_keys = ["conservative", "moderate", "aggressive"]

    for col, scenario_key in enumerate(scenario_keys):
        scenario, results = all_results[scenario_key]
        optimal_rate = find_optimal_rate(results)

        rates = np.array(MAX_RATES)
        # Tracking percentiles
        tracking_median = np.array([results[r]["tracking_median_mean"] for r in rates])
        tracking_p68 = np.array([results[r]["tracking_p68_mean"] for r in rates])
        tracking_p95 = np.array([results[r]["tracking_p95_mean"] for r in rates])
        # Tracking thresholds
        within_05 = np.array([results[r]["within_05_mean"] for r in rates])
        within_10 = np.array([results[r]["within_10_mean"] for r in rates])
        # Gradient percentiles (median, 68th, 95th)
        grad_median = np.array([results[r]["gradient_median_mean"] for r in rates])
        grad_p68 = np.array([results[r]["gradient_p68_mean"] for r in rates])
        grad_p95 = np.array([results[r]["gradient_p95_mean"] for r in rates])

        # Top row: Tracking percentiles
        ax = axes[0, col]

        # Plot median, 68th, 95th percentile
        ax.plot(
            rates,
            tracking_median,
            "o-",
            color="tab:green",
            linewidth=2,
            markersize=8,
            label="Median",
        )
        ax.plot(
            rates,
            tracking_p68,
            "s--",
            color="tab:blue",
            linewidth=2,
            markersize=7,
            label="68th %ile",
        )
        ax.plot(
            rates,
            tracking_p95,
            "^:",
            color="tab:red",
            linewidth=2,
            markersize=7,
            label="95th %ile",
        )

        # Fill between median and 95th
        ax.fill_between(rates, tracking_median, tracking_p95, color="gray", alpha=0.15)

        # Mark optimal
        opt_idx = list(rates).index(optimal_rate)
        ax.axvline(x=optimal_rate, color="red", linestyle=":", alpha=0.7, linewidth=2)
        ax.plot(optimal_rate, tracking_median[opt_idx], "r*", markersize=20, zorder=10)

        ax.set_xlabel("Max Rate (C/h)")
        ax.set_ylabel("Tracking Error (C)")
        ax.set_title(
            f"{scenario.name}\n"
            + r"$\tau_{grad}$"
            + f"={scenario.tau_gradient}h, "
            + r"$\theta$"
            + f"={scenario.gradient_threshold}C",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 2.1)
        ax.set_ylim(0, None)
        ax.legend(loc="upper right", fontsize=9)

        # Bottom row: Gradient percentiles (like tracking panel)
        ax = axes[1, col]

        # Plot median, 68th, 95th percentile gradients
        ax.plot(
            rates,
            grad_median,
            "o-",
            color="tab:green",
            linewidth=2,
            markersize=8,
            label="Median",
        )
        ax.plot(
            rates,
            grad_p68,
            "s--",
            color="tab:blue",
            linewidth=2,
            markersize=7,
            label="68th %ile",
        )
        ax.plot(
            rates,
            grad_p95,
            "^:",
            color="tab:red",
            linewidth=2,
            markersize=7,
            label="95th %ile",
        )

        # Fill between median and 95th
        ax.fill_between(rates, grad_median, grad_p95, color="gray", alpha=0.15)

        # Add threshold reference lines for gradient
        ax.axhline(
            y=0.5,
            color="orange",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        ax.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
        )

        # Mark optimal
        ax.axvline(x=optimal_rate, color="red", linestyle=":", alpha=0.7, linewidth=2)
        opt_grad = grad_median[opt_idx]
        ax.plot(optimal_rate, opt_grad, "r*", markersize=20, zorder=10)

        ax.set_xlabel("Max Rate (C/h)")
        ax.set_ylabel("Internal Gradient (°C)")
        ax.set_title(
            f"Gradient Percentiles (weight={scenario.penalty_weight})", fontsize=11
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 2.1)
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "Gradient Penalty Scenarios: Tracking Percentiles & Thresholds\n"
        "Red star = optimal max_rate for each scenario",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        figures_path / "gradient_scenarios_results.png", dpi=150, bbox_inches="tight"
    )
    print(f"\nSaved: {figures_path / 'gradient_scenarios_results.png'}")
    plt.close()


def create_combined_objective_figure(
    all_results: Dict[str, dict], figures_path: Path
) -> None:
    """Create a figure showing the combined objective for all scenarios."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for scenario_key in ["conservative", "moderate", "aggressive"]:
        scenario, results = all_results[scenario_key]
        optimal_rate = find_optimal_rate(results)

        rates = np.array(MAX_RATES)
        combined = np.array([results[r]["combined_mean"] for r in rates])
        combined_std = np.array([results[r]["combined_std"] for r in rates])

        ax.plot(
            rates,
            combined,
            "o-",
            color=scenario.color,
            linewidth=2.5,
            markersize=10,
            label=scenario.name,
        )
        ax.fill_between(
            rates,
            combined - combined_std,
            combined + combined_std,
            color=scenario.color,
            alpha=0.15,
        )

        # Mark optimal
        opt_idx = list(rates).index(optimal_rate)
        ax.plot(
            optimal_rate,
            combined[opt_idx],
            "*",
            color=scenario.color,
            markersize=25,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    ax.set_xlabel("Max Rate (C/h)", fontsize=12)
    ax.set_ylabel("Combined Objective (lower is better)", fontsize=12)
    ax.set_title(
        "Combined Objective: Tracking RMS + Weighted Gradient Penalty",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 2.1)

    plt.tight_layout()
    plt.savefig(
        figures_path / "gradient_scenarios_combined.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {figures_path / 'gradient_scenarios_combined.png'}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("GRADIENT PENALTY SCENARIOS")
    print("=" * 60)

    # Load data
    print("\nLoading temperature data...")
    df = load_temperature_data()
    days = build_day_database(df)
    _, test_days = train_test_split(days)
    print(f"Loaded {len(test_days)} test nights")

    figures_path = get_figures_path()
    figures_path.mkdir(exist_ok=True)

    # Run scenarios
    all_results = {}

    for scenario_key, scenario in SCENARIOS.items():
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  tau_gradient = {scenario.tau_gradient}h")
        print(f"  gradient_threshold = {scenario.gradient_threshold}C")
        print(f"  penalty_weight = {scenario.penalty_weight}")
        print("=" * 60)

        results = run_sweep(test_days, scenario, MAX_RATES)
        all_results[scenario_key] = (scenario, results)

        optimal_rate = find_optimal_rate(results)
        print(f"\n  Optimal max_rate: {optimal_rate:.2f} C/h")

    # Print summary
    print_summary_table(all_results)

    # Create figures
    print("\nGenerating figures...")
    create_figure(all_results, figures_path)
    create_combined_objective_figure(all_results, figures_path)

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMAL RATES BY SCENARIO")
    print("=" * 60)
    for scenario_key in ["conservative", "moderate", "aggressive"]:
        scenario, results = all_results[scenario_key]
        optimal_rate = find_optimal_rate(results)
        opt_metrics = results[optimal_rate]
        print(f"\n{scenario.name}:")
        print(f"  Optimal rate: {optimal_rate:.2f} C/h")
        print(
            f"  Tracking: median={opt_metrics['tracking_median_mean']:.3f}C, "
            f"68%ile={opt_metrics['tracking_p68_mean']:.3f}C, "
            f"95%ile={opt_metrics['tracking_p95_mean']:.3f}C"
        )
        print(
            f"  Thresholds: {opt_metrics['within_05_mean']:.1f}% <0.5C, "
            f"{opt_metrics['within_10_mean']:.1f}% <1.0C"
        )
        print(f"  Gradient 95th %ile: {opt_metrics['gradient_p95_mean']:.3f}C")


if __name__ == "__main__":
    main()
