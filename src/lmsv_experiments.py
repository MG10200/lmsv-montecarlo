"""
Numerical Experiments for the LMSV Monte Carlo Simulation

Master's Thesis:
"Long-Memory Stochastic Volatility Models and Fractional Delta Hedging"

Author:
    Marco Guidi

This script reproduces the numerical experiments reported in the thesis.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmsv_monte_carlo import (
    LMSVParameters,
    black_scholes_call,
    black_scholes_delta,
    estimate_lmsv_quantities,
    run_lmsv_monte_carlo,
    validate_black_scholes_limit,
)

OUTPUT_DIR = Path("lmsv_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_csv(filename, rows):
    if not rows:
        return
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def monte_carlo_convergence_study(
    params,
    number_of_steps=252,
    path_counts=(1_000, 5_000, 10_000, 25_000, 50_000),
    batch_size=5_000,
    seed=20260719,
):
    rows = []
    for paths in path_counts:
        result, _ = run_lmsv_monte_carlo(
            params=params,
            number_of_steps=number_of_steps,
            number_of_paths=paths,
            batch_size=min(batch_size, paths),
            seed=seed,
        )
        rows.append({
            "paths": paths,
            "steps": number_of_steps,
            "price": result["price"],
            "price_se": result["price_se"],
            "price_ci_lower": result["price"] - 1.96 * result["price_se"],
            "price_ci_upper": result["price"] + 1.96 * result["price_se"],
            "delta": result["delta"],
            "delta_se": result["delta_se"],
            "delta_ci_lower": result["delta"] - 1.96 * result["delta_se"],
            "delta_ci_upper": result["delta"] + 1.96 * result["delta_se"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
        print(
            f"paths={paths:>7,} | price={result['price']:.8f} | "
            f"SE={result['price_se']:.3e} | delta={result['delta']:.8f}"
        )

    save_csv("monte_carlo_convergence.csv", rows)

    n = np.array([r["paths"] for r in rows], dtype=float)
    se = np.array([r["price_se"] for r in rows], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.loglog(n, se, marker="o", label="Observed price SE")
    plt.loglog(n, se[0] * np.sqrt(n[0] / n), linestyle="--",
               label=r"$N^{-1/2}$ reference")
    plt.xlabel("Number of Monte Carlo paths")
    plt.ylabel("Standard error of call price")
    plt.title("Monte Carlo convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monte_carlo_convergence.png", dpi=200)
    plt.close()
    return rows


def time_grid_stability_study(
    params,
    step_counts=(32, 64, 128, 252, 512),
    number_of_paths=25_000,
    batch_size=5_000,
    seed=20260719,
):
    rows = []
    for steps in step_counts:
        result, _ = run_lmsv_monte_carlo(
            params=params,
            number_of_steps=steps,
            number_of_paths=number_of_paths,
            batch_size=batch_size,
            seed=seed,
        )
        rows.append({
            "paths": number_of_paths,
            "steps": steps,
            "price": result["price"],
            "price_se": result["price_se"],
            "delta": result["delta"],
            "delta_se": result["delta_se"],
            "mean_integrated_volatility": result["integrated_vol_mean"],
            "std_integrated_volatility": result["integrated_vol_std"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
        print(
            f"steps={steps:>4} | price={result['price']:.8f} | "
            f"delta={result['delta']:.8f} | E[U]={result['integrated_vol_mean']:.8f}"
        )

    save_csv("time_grid_stability.csv", rows)

    steps = np.array([r["steps"] for r in rows])
    price = np.array([r["price"] for r in rows])
    se = np.array([r["price_se"] for r in rows])

    plt.figure(figsize=(8, 5))
    plt.errorbar(steps, price, yerr=1.96 * se, marker="o", capsize=4)
    plt.xlabel("Number of time steps")
    plt.ylabel("Estimated call price")
    plt.title("Time-grid stability")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_grid_stability.png", dpi=200)
    plt.close()
    return rows


def strike_bias_study(
    params,
    strikes=(80, 90, 95, 100, 105, 110, 120),
    number_of_steps=252,
    number_of_paths=50_000,
    batch_size=5_000,
    seed=20260719,
):
    _, integrated_volatility = run_lmsv_monte_carlo(
        params=params,
        number_of_steps=number_of_steps,
        number_of_paths=number_of_paths,
        batch_size=batch_size,
        seed=seed,
    )

    rows = []
    forward = params.spot * np.exp(params.rate * params.maturity)

    for strike in strikes:
        p = replace(params, strike=float(strike))
        result = estimate_lmsv_quantities(p, integrated_volatility)
        m = np.log(p.spot / (p.strike * np.exp(-p.rate * p.maturity)))

        rows.append({
            "strike": strike,
            "forward": forward,
            "adjusted_log_moneyness": m,
            "lmsv_price": result["price"],
            "price_se": result["price_se"],
            "lmsv_delta": result["delta"],
            "delta_se": result["delta_se"],
            "implied_volatility": result["implied_volatility"],
            "bs_implied_vol_delta": result["black_scholes_delta"],
            "hedging_bias": result["hedging_bias"],
        })
        print(f"K={strike:>6.2f} | m={m:+.6f} | bias={result['hedging_bias']:+.6e}")

    save_csv("strike_hedging_bias.csv", rows)

    x = np.array([r["strike"] for r in rows])
    y = np.array([r["hedging_bias"] for r in rows])

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(forward, linestyle="--", label=r"ATM-forward strike $S_0e^{rT}$")
    plt.plot(x, y, marker="o")
    plt.xlabel("Strike")
    plt.ylabel(r"Hedging bias $\Delta^{BS}(\sigma^i)-\Delta^{LMSV}$")
    plt.title("Hedging bias across strikes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strike_hedging_bias.png", dpi=200)
    plt.close()
    return rows


def hurst_sensitivity_study(
    params,
    hurst_values=(0.55, 0.65, 0.75, 0.85, 0.95),
    number_of_steps=252,
    number_of_paths=25_000,
    batch_size=5_000,
    seed=20260719,
):
    rows = []
    for h in hurst_values:
        p = replace(params, hurst=float(h))
        result, _ = run_lmsv_monte_carlo(
            params=p,
            number_of_steps=number_of_steps,
            number_of_paths=number_of_paths,
            batch_size=batch_size,
            seed=seed,
        )
        rows.append({
            "hurst": h,
            "price": result["price"],
            "price_se": result["price_se"],
            "delta": result["delta"],
            "delta_se": result["delta_se"],
            "implied_volatility": result["implied_volatility"],
            "hedging_bias": result["hedging_bias"],
            "mean_integrated_volatility": result["integrated_vol_mean"],
            "std_integrated_volatility": result["integrated_vol_std"],
        })
        print(
            f"H={h:.2f} | price={result['price']:.8f} | "
            f"std(U)={result['integrated_vol_std']:.8f}"
        )

    save_csv("hurst_sensitivity.csv", rows)

    x = np.array([r["hurst"] for r in rows])
    y = np.array([r["std_integrated_volatility"] for r in rows])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Hurst parameter")
    plt.ylabel(r"Standard deviation of $U_{0,T}$")
    plt.title("Sensitivity to the Hurst parameter")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hurst_sensitivity.png", dpi=200)
    plt.close()
    return rows


def main():
    params = LMSVParameters()

    print("\n1. BLACK--SCHOLES LIMIT")
    validate_black_scholes_limit(params)

    sigma0 = 0.5 * (params.sigma_min + params.sigma_max)
    print("Benchmark price:",
          black_scholes_call(params.spot, params.strike, params.maturity,
                             params.rate, sigma0))
    print("Benchmark Delta:",
          black_scholes_delta(params.spot, params.strike, params.maturity,
                              params.rate, sigma0))

    print("\n2. MONTE CARLO CONVERGENCE")
    monte_carlo_convergence_study(params)

    print("\n3. TIME-GRID STABILITY")
    time_grid_stability_study(params)

    print("\n4. HEDGING BIAS ACROSS STRIKES")
    strike_bias_study(params)

    print("\n5. HURST SENSITIVITY")
    hurst_sensitivity_study(params)

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
