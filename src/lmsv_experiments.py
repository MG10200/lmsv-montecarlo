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
    estimate_asian_quantities,
    run_lmsv_monte_carlo,
    run_lmsv_asian_monte_carlo,
    validate_black_scholes_limit,
)

OUTPUT_DIR = Path("results")
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
    plt.grid(True, alpha=0.25, linewidth=0.7)
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
    plt.errorbar(steps, price, yerr=1.96 * se, marker="o", capsize=4, linewidth=1.8, elinewidth=1.1, capthick=1.1,)    
    plt.xlabel("Number of time steps")
    plt.ylabel("Estimated call price")
    plt.title("Time-grid stability")
    plt.grid(True, alpha=0.25, linewidth=0.7)
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
    plt.grid(True, alpha=0.25, linewidth=0.7)
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
    plt.grid(True, alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hurst_sensitivity.png", dpi=200)
    plt.close()
    return rows


def asian_baseline_study(
    params,
    number_of_steps=252,
    number_of_paths=50_000,
    batch_size=5_000,
    seed=20260719,
    spot_bump=0.10,
):
    """
    Run the baseline arithmetic Asian call experiment.

    The continuously monitored arithmetic average is approximated
    through the trapezoidal rule on the numerical time grid.
    """

    result, average_ratio = run_lmsv_asian_monte_carlo(
        params=params,
        number_of_steps=number_of_steps,
        number_of_paths=number_of_paths,
        batch_size=batch_size,
        seed=seed,
        spot_bump=spot_bump,
    )

    rows = [{
        "paths": number_of_paths,
        "steps": number_of_steps,
        "price": result["price"],
        "price_se": result["price_se"],
        "mean_payoff": result["mean_payoff"],
        "delta_pathwise": result["delta_pathwise"],
        "delta_pathwise_se": result["delta_pathwise_se"],
        "delta_finite_difference":
            result["delta_finite_difference"],
        "delta_finite_difference_se":
            result["delta_finite_difference_se"],
        "delta_difference":
            result["delta_difference"],
        "delta_difference_se":
            result["delta_difference_se"],
        "spot_bump": result["spot_bump"],
        "average_price_mean":
            result["average_price_mean"],
        "average_price_std":
            result["average_price_std"],
        "elapsed_seconds":
            result["elapsed_seconds"],
    }]

    save_csv(
        "asian_baseline.csv",
        rows,
    )

    print(
        f"price={result['price']:.8f} | "
        f"PW Delta={result['delta_pathwise']:.8f} | "
        f"FD Delta={result['delta_finite_difference']:.8f} | "
        f"difference={result['delta_difference']:+.3e}"
    )

    return result, average_ratio


def asian_delta_bump_study(
    params,
    average_ratio,
    bump_values=(0.05, 0.10, 0.25, 0.50),
):
    """
    Compare the pathwise Asian Delta with centered finite differences
    for different spot bumps using the same Monte Carlo sample.
    """

    rows = []

    for bump in bump_values:
        result = estimate_asian_quantities(
            params=params,
            average_ratio=average_ratio,
            spot_bump=float(bump),
        )

        if result["delta_difference_se"] > 0.0:
            z_score = (
                result["delta_difference"]
                / result["delta_difference_se"]
            )
        else:
            z_score = np.nan

        rows.append({
            "spot_bump": bump,
            "delta_pathwise":
                result["delta_pathwise"],
            "delta_pathwise_se":
                result["delta_pathwise_se"],
            "delta_finite_difference":
                result["delta_finite_difference"],
            "delta_finite_difference_se":
                result["delta_finite_difference_se"],
            "delta_difference":
                result["delta_difference"],
            "delta_difference_se":
                result["delta_difference_se"],
            "difference_z_score":
                z_score,
        })

        print(
            f"bump={bump:.2f} | "
            f"PW={result['delta_pathwise']:.8f} | "
            f"FD={result['delta_finite_difference']:.8f} | "
            f"difference={result['delta_difference']:+.6e} | "
            f"SE(diff)={result['delta_difference_se']:.3e}"
        )

    save_csv(
        "asian_delta_bump_sensitivity.csv",
        rows,
    )

    x = np.array(
        [r["spot_bump"] for r in rows],
        dtype=float,
    )

    fd_delta = np.array(
        [r["delta_finite_difference"] for r in rows],
        dtype=float,
    )

    pw_delta = rows[0]["delta_pathwise"]

    plt.figure(figsize=(8, 5))
    plt.axhline(
        pw_delta,
        linestyle="--",
        color="firebrick",
        linewidth=1.6,
        label="Pathwise Delta",
    )

    plt.plot(
        x,
        fd_delta,
        marker="o",
        color="firebrick",
        linewidth=1.6,
        label="Finite-difference Delta",
    )

    plt.xlabel("Spot perturbation $h$")
    plt.ylabel("Asian call Delta")
    plt.title(
        "Asian Delta: Pathwise and Finite-Difference Estimators"
    )
    plt.legend()
    plt.grid(True, alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "asian_delta_bump_sensitivity.png",
        dpi=200,
    )
    plt.close()

    return rows


def asian_time_grid_stability_study(
    params,
    step_counts=(32, 64, 128, 252, 512),
    number_of_paths=25_000,
    batch_size=5_000,
    seed=20260719,
    spot_bump=0.10,
):
    """
    Study the stability of the arithmetic Asian call price
    and pathwise Delta with respect to the time-grid resolution.
    """

    rows = []

    for steps in step_counts:
        result, _ = run_lmsv_asian_monte_carlo(
            params=params,
            number_of_steps=steps,
            number_of_paths=number_of_paths,
            batch_size=min(
                batch_size,
                number_of_paths,
            ),
            seed=seed,
            spot_bump=spot_bump,
        )

        rows.append({
            "paths": number_of_paths,
            "steps": steps,
            "price": result["price"],
            "price_se": result["price_se"],
            "delta_pathwise":
                result["delta_pathwise"],
            "delta_pathwise_se":
                result["delta_pathwise_se"],
            "average_price_mean":
                result["average_price_mean"],
            "average_price_std":
                result["average_price_std"],
            "elapsed_seconds":
                result["elapsed_seconds"],
        })

        print(
            f"steps={steps:>4} | "
            f"price={result['price']:.8f} | "
            f"SE={result['price_se']:.3e} | "
            f"PW Delta={result['delta_pathwise']:.8f} | "
            f"Delta SE={result['delta_pathwise_se']:.3e} | "
            f"E[A]={result['average_price_mean']:.8f}"
        )

    save_csv(
        "asian_time_grid_stability.csv",
        rows,
    )

    steps = np.array(
        [r["steps"] for r in rows]
    )

    price = np.array(
        [r["price"] for r in rows]
    )

    price_se = np.array(
        [r["price_se"] for r in rows]
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        steps,
        price,
        yerr=1.96 * price_se,
        marker="o",
        capsize=4,
        color="firebrick",
        linewidth=1.8,
        elinewidth=1.1,
        capthick=1.1,
    )
    plt.xlabel("Number of time steps")
    plt.ylabel("Estimated Asian call price")
    plt.title(
        "Arithmetic Asian call: time-grid stability"
    )
    plt.grid(True, alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "asian_time_grid_stability.png",
        dpi=200,
    )
    plt.close()

    return rows


def european_asian_hurst_comparison_study(
    params,
    european_rows,
    number_of_steps=252,
    number_of_paths=25_000,
    batch_size=5_000,
    seed=20260719,
    spot_bump=0.10,
):
    """
    Compare European and arithmetic Asian call prices and Deltas
    across the same Hurst-parameter values.

    The European results are reused from the existing Hurst
    sensitivity study.
    """

    rows = []

    for european_row in european_rows:
        h = float(
            european_row["hurst"]
        )

        asian_params = replace(
            params,
            hurst=h,
        )

        asian_result, _ = run_lmsv_asian_monte_carlo(
            params=asian_params,
            number_of_steps=number_of_steps,
            number_of_paths=number_of_paths,
            batch_size=batch_size,
            seed=seed,
            spot_bump=spot_bump,
        )

        rows.append({
            "hurst": h,

            "european_price":
                european_row["price"],
            "european_price_se":
                european_row["price_se"],

            "asian_price":
                asian_result["price"],
            "asian_price_se":
                asian_result["price_se"],

            "price_difference":
                european_row["price"]
                - asian_result["price"],

            "european_delta":
                european_row["delta"],
            "european_delta_se":
                european_row["delta_se"],

            "asian_delta":
                asian_result["delta_pathwise"],
            "asian_delta_se":
                asian_result["delta_pathwise_se"],

            "delta_difference":
                european_row["delta"]
                - asian_result["delta_pathwise"],

            "asian_average_price_mean":
                asian_result["average_price_mean"],

            "asian_average_price_std":
                asian_result["average_price_std"],

            "price_difference_over_se":
                (
                    european_row["price"]
                    - asian_result["price"]
                )
                / asian_result["price_se"],

            "delta_difference_over_se":
                (
                    european_row["delta"]
                    - asian_result["delta_pathwise"]
                )
                / asian_result["delta_pathwise_se"],
        })

        print(
            f"H={h:.2f} | "
            f"European price={european_row['price']:.8f} "
            f"(SE={european_row['price_se']:.3e}) | "
            f"Asian price={asian_result['price']:.8f} "
            f"(SE={asian_result['price_se']:.3e}) | "
            f"European Delta={european_row['delta']:.8f} "
            f"(SE={european_row['delta_se']:.3e}) | "
            f"Asian Delta={asian_result['delta_pathwise']:.8f} "
            f"(SE={asian_result['delta_pathwise_se']:.3e})"
        )

    save_csv(
        "european_asian_hurst_comparison.csv",
        rows,
    )

    h_values = np.array(
        [r["hurst"] for r in rows]
    )

    european_price = np.array(
        [r["european_price"] for r in rows]
    )

    asian_price = np.array(
        [r["asian_price"] for r in rows]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        h_values,
        european_price,
        marker="o",
        color="tab:blue",
        label="European call",
    )

    plt.plot(
        h_values,
        asian_price,
        marker="o",
        color="firebrick",
        label="Arithmetic Asian call",
    )

    plt.xlabel("Hurst parameter")
    plt.ylabel("Option price")
    plt.title(
        "European and Asian call prices across H"
    )
    plt.legend()
    plt.grid(True, alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "european_asian_price_hurst.png",
        dpi=200,
    )
    plt.close()

    european_delta = np.array(
        [r["european_delta"] for r in rows]
    )

    asian_delta = np.array(
        [r["asian_delta"] for r in rows]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        h_values,
        european_delta,
        marker="o",
        color="tab:blue",
        label="European call",
    )

    plt.plot(
        h_values,
        asian_delta,
        marker="o",
        color="firebrick",
        label="Arithmetic Asian call",
    )

    plt.xlabel("Hurst parameter")
    plt.ylabel("Delta")
    plt.title(
        "European and Asian call Deltas across H"
    )
    plt.legend()
    plt.grid(True, alpha=0.25, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "european_asian_delta_hurst.png",
        dpi=200,
    )
    plt.close()

    return rows





def main():
    params = LMSVParameters()

    # ======================================================
    # European call experiments
    # ======================================================

    print("\n1. BLACK--SCHOLES LIMIT")
    validate_black_scholes_limit(params)

    sigma0 = 0.5 * (
        params.sigma_min
        + params.sigma_max
    )

    print(
        "Benchmark price:",
        black_scholes_call(
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            sigma0,
        ),
    )

    print(
        "Benchmark Delta:",
        black_scholes_delta(
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            sigma0,
        ),
    )

    print("\n2. MONTE CARLO CONVERGENCE")
    monte_carlo_convergence_study(
        params
    )

    print("\n3. TIME-GRID STABILITY")
    time_grid_stability_study(
        params
    )

    print("\n4. HEDGING BIAS ACROSS STRIKES")
    strike_bias_study(
        params
    )

    print("\n5. HURST SENSITIVITY")
    european_hurst_rows = (
        hurst_sensitivity_study(
            params
        )
    )

    # ======================================================
    # Arithmetic Asian call experiments
    # ======================================================

    print(
        "\n=========================================="
    )
    print(
        "ARITHMETIC ASIAN OPTION"
    )
    print(
        "=========================================="
    )

    print("\n6. ASIAN CALL BASELINE")
    asian_result, asian_average_ratio = (
        asian_baseline_study(
            params
        )
    )

    print(
        "\n7. ASIAN DELTA ESTIMATOR COMPARISON"
    )
    asian_delta_bump_study(
        params=params,
        average_ratio=asian_average_ratio,
    )

    print(
        "\n8. ASIAN TIME-GRID STABILITY"
    )
    asian_time_grid_stability_study(
        params
    )

    print(
        "\n9. EUROPEAN VS ASIAN: "
        "HURST SENSITIVITY"
    )
    european_asian_hurst_comparison_study(
        params=params,
        european_rows=european_hurst_rows,
    )

    print(
        f"\nAll outputs saved in: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()