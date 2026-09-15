"""
LMSV Monte Carlo simulation at time t=0.

Requirements:
    numpy
    scipy
    matplotlib

Run:
    python lmsv_monte_carlo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import brentq
from scipy.special import ndtr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class LMSVParameters:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.02
    hurst: float = 0.70
    mean_reversion: float = 2.0
    vol_of_vol: float = 0.35
    initial_factor: float = 0.0
    sigma_min: float = 0.15
    sigma_max: float = 0.30
    sigma_slope: float = 1.0


def logistic_volatility(y, sigma_min, sigma_max, slope):
    logistic = 0.5 * (1.0 + np.tanh(0.5 * slope * y))
    return sigma_min + (sigma_max - sigma_min) * logistic


def fractional_gaussian_noise_covariance(number_of_steps, dt, hurst):
    index = np.arange(number_of_steps)
    lag = np.abs(index[:, None] - index[None, :]).astype(float)
    gamma = 0.5 * (
        np.abs(lag + 1.0) ** (2.0 * hurst)
        - 2.0 * lag ** (2.0 * hurst)
        + np.abs(lag - 1.0) ** (2.0 * hurst)
    )
    return (dt ** (2.0 * hurst)) * gamma


def build_fgn_cholesky(number_of_steps, dt, hurst):
    covariance = fractional_gaussian_noise_covariance(
        number_of_steps, dt, hurst
    )
    jitter = 1e-13 * max(1.0, float(np.max(np.diag(covariance))))
    covariance = covariance + jitter * np.eye(number_of_steps)
    return cholesky(covariance, lower=True, check_finite=False)


def black_scholes_call(spot, strike, maturity, rate, volatility):
    if volatility <= 0.0:
        return max(spot - strike * exp(-rate * maturity), 0.0)

    total_volatility = volatility * sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / total_volatility
    d2 = d1 - total_volatility
    return spot * float(ndtr(d1)) - strike * exp(-rate * maturity) * float(ndtr(d2))


def black_scholes_delta(spot, strike, maturity, rate, volatility):
    if volatility <= 0.0:
        return float(spot > strike * exp(-rate * maturity))

    d1 = (
        log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt(maturity))
    return float(ndtr(d1))


def implied_volatility_from_call(target_price, spot, strike, maturity, rate):
    lower_bound = max(spot - strike * exp(-rate * maturity), 0.0)
    upper_bound = spot

    if not lower_bound - 1e-10 <= target_price <= upper_bound + 1e-10:
        raise ValueError("Estimated call price violates no-arbitrage bounds.")

    if abs(target_price - lower_bound) <= 1e-10:
        return 0.0

    def objective(vol):
        return black_scholes_call(
            spot, strike, maturity, rate, vol
        ) - target_price

    return float(brentq(objective, 1e-10, 5.0, xtol=1e-12, rtol=1e-12))


def simulate_integrated_volatility(
    params,
    number_of_steps=252,
    number_of_paths=50000,
    batch_size=5000,
    seed=20260719,
):
    if not 0.5 < params.hurst < 1.0:
        raise ValueError("hurst must lie in (0.5, 1).")
    if params.sigma_max > 2.0 * params.sigma_min:
        raise ValueError("Require sigma_max <= 2*sigma_min.")

    dt = params.maturity / number_of_steps
    decay = exp(-params.mean_reversion * dt)
    cholesky_factor = build_fgn_cholesky(
        number_of_steps, dt, params.hurst
    )

    rng = np.random.default_rng(seed)
    integrated_volatility = np.empty(number_of_paths, dtype=float)

    start = 0
    while start < number_of_paths:
        current_batch = min(batch_size, number_of_paths - start)

        independent_normals = rng.standard_normal(
            size=(number_of_steps, current_batch)
        )
        fgn_increments = cholesky_factor @ independent_normals

        factor = np.full(current_batch, params.initial_factor, dtype=float)
        integrated_variance = np.zeros(current_batch, dtype=float)

        for step in range(number_of_steps):
            sigma = logistic_volatility(
                factor,
                params.sigma_min,
                params.sigma_max,
                params.sigma_slope,
            )
            integrated_variance += sigma * sigma * dt

            factor = (
                decay * factor
                + params.vol_of_vol * decay * fgn_increments[step]
            )

        integrated_volatility[start:start + current_batch] = np.sqrt(
            integrated_variance
        )
        start += current_batch

    return integrated_volatility


def estimate_lmsv_quantities(params, integrated_volatility):
    discount = exp(-params.rate * params.maturity)
    m = log(params.spot / (params.strike * discount))

    d1 = m / integrated_volatility + 0.5 * integrated_volatility
    d2 = d1 - integrated_volatility

    conditional_deltas = ndtr(d1)
    conditional_prices = (
        params.spot * ndtr(d1)
        - params.strike * discount * ndtr(d2)
    )

    n = integrated_volatility.size
    price = float(np.mean(conditional_prices))
    price_se = float(np.std(conditional_prices, ddof=1) / sqrt(n))
    delta = float(np.mean(conditional_deltas))
    delta_se = float(np.std(conditional_deltas, ddof=1) / sqrt(n))

    implied_vol = implied_volatility_from_call(
        price,
        params.spot,
        params.strike,
        params.maturity,
        params.rate,
    )
    bs_delta = black_scholes_delta(
        params.spot,
        params.strike,
        params.maturity,
        params.rate,
        implied_vol,
    )

    return {
        "price": price,
        "price_se": price_se,
        "delta": delta,
        "delta_se": delta_se,
        "implied_volatility": implied_vol,
        "black_scholes_delta": bs_delta,
        "hedging_bias": bs_delta - delta,
        "integrated_vol_mean": float(np.mean(integrated_volatility)),
        "integrated_vol_std": float(np.std(integrated_volatility, ddof=1)),
    }


def run_lmsv_monte_carlo(
    params,
    number_of_steps=252,
    number_of_paths=50000,
    batch_size=5000,
    seed=20260719,
):
    start = perf_counter()
    integrated_volatility = simulate_integrated_volatility(
        params=params,
        number_of_steps=number_of_steps,
        number_of_paths=number_of_paths,
        batch_size=batch_size,
        seed=seed,
    )
    result = estimate_lmsv_quantities(params, integrated_volatility)
    result["elapsed_seconds"] = perf_counter() - start
    result["number_of_steps"] = number_of_steps
    result["number_of_paths"] = number_of_paths
    return result, integrated_volatility




# ==========================================================
# Arithmetic Asian call under the LMSV model
# ==========================================================

def simulate_asian_average_ratios(
    params,
    number_of_steps=252,
    number_of_paths=50000,
    batch_size=5000,
    seed=20260719,
):
    """
    Simulate the time average of S_t / S_0 for a continuously monitored
    arithmetic Asian option.

    The continuous average

        A_T = (1/T) * integral_0^T S_t dt

    is approximated through the trapezoidal rule.

    Only the average ratio A_T / S_0 is retained. Complete stock-price
    and volatility-factor trajectories are not stored.
    """

    if not 0.5 < params.hurst < 1.0:
        raise ValueError("hurst must lie in (0.5, 1).")

    if params.sigma_max > 2.0 * params.sigma_min:
        raise ValueError("Require sigma_max <= 2*sigma_min.")

    dt = params.maturity / number_of_steps
    sqrt_dt = sqrt(dt)

    decay = exp(-params.mean_reversion * dt)

    cholesky_factor = build_fgn_cholesky(
        number_of_steps,
        dt,
        params.hurst,
    )

    # Two independent pseudo-random streams are used to make explicit
    # the independence between the fractional volatility noise and the
    # Brownian motion driving the asset price.
    seed_sequence = np.random.SeedSequence(seed)
    fgn_seed, stock_seed = seed_sequence.spawn(2)

    rng_fgn = np.random.default_rng(fgn_seed)
    rng_stock = np.random.default_rng(stock_seed)

    average_ratio = np.empty(number_of_paths, dtype=float)

    start = 0

    while start < number_of_paths:

        current_batch = min(
            batch_size,
            number_of_paths - start,
        )

        # Fractional Gaussian-noise increments for the volatility factor.
        independent_normals = rng_fgn.standard_normal(
            size=(number_of_steps, current_batch)
        )

        fgn_increments = (
            cholesky_factor @ independent_normals
        )

        factor = np.full(
            current_batch,
            params.initial_factor,
            dtype=float,
        )

        # Work with S_t / S_0 rather than S_t itself.
        # Since the LMSV stock dynamics are multiplicative,
        # this ratio does not depend on the numerical value of S_0.
        stock_ratio = np.ones(
            current_batch,
            dtype=float,
        )

        integrated_stock_ratio = np.zeros(
            current_batch,
            dtype=float,
        )

        for step in range(number_of_steps):

            sigma = logistic_volatility(
                factor,
                params.sigma_min,
                params.sigma_max,
                params.sigma_slope,
            )

            # Brownian shocks driving the stock price.
            stock_normals = rng_stock.standard_normal(
                current_batch
            )

            next_stock_ratio = stock_ratio * np.exp(
                (
                    params.rate
                    - 0.5 * sigma * sigma
                )
                * dt
                + sigma
                * sqrt_dt
                * stock_normals
            )

            # Trapezoidal approximation of
            #
            # integral_0^T (S_t / S_0) dt.
            integrated_stock_ratio += (
                0.5
                * (stock_ratio + next_stock_ratio)
                * dt
            )

            stock_ratio = next_stock_ratio

            # Fractional OU update.
            factor = (
                decay * factor
                + params.vol_of_vol
                * decay
                * fgn_increments[step]
            )

        average_ratio[
            start:start + current_batch
        ] = (
            integrated_stock_ratio
            / params.maturity
        )

        start += current_batch

    return average_ratio


def estimate_asian_quantities(
    params,
    average_ratio,
    spot_bump=0.1,
):
    """
    Estimate the arithmetic Asian call price and its initial Delta.

    Delta is computed using both:

        1. the pathwise derivative estimator;
        2. a centered finite-difference estimator using common
           random numbers.
    """

    if spot_bump <= 0.0:
        raise ValueError("spot_bump must be positive.")

    if spot_bump >= params.spot:
        raise ValueError(
            "spot_bump must be smaller than the initial spot."
        )

    discount = exp(
        -params.rate * params.maturity
    )

    # ------------------------------------------------------
    # Asian option price
    # ------------------------------------------------------

    average_price = (
        params.spot * average_ratio
    )

    payoffs = np.maximum(
        average_price - params.strike,
        0.0,
    )

    discounted_payoffs = (
        discount * payoffs
    )

    n = average_ratio.size

    # Mean undiscounted payoff. This quantity is retained mainly
    # as a diagnostic check; the option price is the discounted
    # expectation computed below.

    mean_payoff = float(
        np.mean(payoffs)
    )

    price = float(
        np.mean(discounted_payoffs)
    )

    price_se = float(
        np.std(
            discounted_payoffs,
            ddof=1,
        )
        / sqrt(n)
    )

    # ------------------------------------------------------
    # Pathwise Delta
    # ------------------------------------------------------
    #
    # Since
    #
    # A_T = S_0 * average_ratio,
    #
    # we have
    #
    # dA_T / dS_0 = average_ratio.
    #
    # Hence
    #
    # Delta =
    # e^{-rT} E[
    #     1_{A_T > K}
    #     average_ratio
    # ].
    # ------------------------------------------------------

    pathwise_delta_samples = (
        discount
        * (average_price > params.strike)
        * average_ratio
    )

    delta_pathwise = float(
        np.mean(pathwise_delta_samples)
    )

    delta_pathwise_se = float(
        np.std(
            pathwise_delta_samples,
            ddof=1,
        )
        / sqrt(n)
    )

    # ------------------------------------------------------
    # Finite-difference Delta
    # ------------------------------------------------------
    #
    # The same simulated random paths are used for S_0+h
    # and S_0-h. This is a common-random-numbers variance
    # reduction technique.
    # ------------------------------------------------------

    spot_plus = (
        params.spot + spot_bump
    )

    spot_minus = (
        params.spot - spot_bump
    )

    payoff_plus = np.maximum(
        spot_plus * average_ratio
        - params.strike,
        0.0,
    )

    payoff_minus = np.maximum(
        spot_minus * average_ratio
        - params.strike,
        0.0,
    )

    finite_difference_samples = (
        discount
        * (
            payoff_plus
            - payoff_minus
        )
        / (2.0 * spot_bump)
    )

    delta_finite_difference = float(
        np.mean(
            finite_difference_samples
        )
    )

    delta_finite_difference_se = float(
        np.std(
            finite_difference_samples,
            ddof=1,
        )
        / sqrt(n)
    )

    # ------------------------------------------------------
    # Difference between the two Delta estimators
    # ------------------------------------------------------
    #
    # Since both estimators are computed from the same
    # simulated paths, their difference should be analysed
    # path by path rather than by combining their standard
    # errors as if they were independent.
    # ------------------------------------------------------

    delta_difference_samples = (
        finite_difference_samples
        - pathwise_delta_samples
    )

    delta_difference = float(
        np.mean(delta_difference_samples)
    )

    delta_difference_se = float(
        np.std(
            delta_difference_samples,
            ddof=1,
        )
        / sqrt(n)
    )


    return {
        "price": price,
        "price_se": price_se,
        "mean_payoff": mean_payoff,
        "delta_pathwise": delta_pathwise,
        "delta_pathwise_se": delta_pathwise_se,
        "delta_finite_difference":
            delta_finite_difference,
        "delta_finite_difference_se":
            delta_finite_difference_se,
        "delta_difference":
            delta_difference,
        "delta_difference_se":
            delta_difference_se,
        "spot_bump": spot_bump,
        "average_price_mean": float(
            np.mean(average_price)
        ),
        "average_price_std": float(
            np.std(
                average_price,
                ddof=1,
            )
        ),
    }


def run_lmsv_asian_monte_carlo(
    params,
    number_of_steps=252,
    number_of_paths=50000,
    batch_size=5000,
    seed=20260719,
    spot_bump=0.1,
):
    """
    Run the complete LMSV Monte Carlo experiment for an
    arithmetic Asian call option.
    """

    start = perf_counter()

    average_ratio = (
        simulate_asian_average_ratios(
            params=params,
            number_of_steps=number_of_steps,
            number_of_paths=number_of_paths,
            batch_size=batch_size,
            seed=seed,
        )
    )

    result = estimate_asian_quantities(
        params=params,
        average_ratio=average_ratio,
        spot_bump=spot_bump,
    )

    result["elapsed_seconds"] = (
        perf_counter() - start
    )

    result["number_of_steps"] = (
        number_of_steps
    )

    result["number_of_paths"] = (
        number_of_paths
    )

    return result, average_ratio






def validate_black_scholes_limit(params):
    deterministic = LMSVParameters(
        spot=params.spot,
        strike=params.strike,
        maturity=params.maturity,
        rate=params.rate,
        hurst=params.hurst,
        mean_reversion=params.mean_reversion,
        vol_of_vol=0.0,
        initial_factor=0.0,
        sigma_min=params.sigma_min,
        sigma_max=params.sigma_max,
        sigma_slope=params.sigma_slope,
    )

    result, _ = run_lmsv_monte_carlo(
        deterministic,
        number_of_steps=252,
        number_of_paths=10000,
        batch_size=5000,
    )

    constant_vol = 0.5 * (params.sigma_min + params.sigma_max)
    bs_price = black_scholes_call(
        params.spot, params.strike, params.maturity, params.rate, constant_vol
    )
    bs_delta = black_scholes_delta(
        params.spot, params.strike, params.maturity, params.rate, constant_vol
    )

    print("\nBLACK--SCHOLES LIMIT CHECK")
    print(f"LMSV price: {result['price']:.12f}")
    print(f"BS price:   {bs_price:.12f}")
    print(f"LMSV Delta: {result['delta']:.12f}")
    print(f"BS Delta:   {bs_delta:.12f}")


def main():
    params = LMSVParameters()

    validate_black_scholes_limit(params)

    result, integrated_volatility = run_lmsv_monte_carlo(params)

    print("\nBASELINE LMSV RESULTS")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")

    plt.figure(figsize=(8, 5))
    plt.hist(integrated_volatility, bins=50, density=True)
    plt.xlabel(r"Integrated volatility $U_{0,T}$")
    plt.ylabel("Density")
    plt.title("Distribution of simulated integrated volatility")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "integrated_volatility_histogram.png",
        dpi=200,
    )
    plt.close()

    print(
        "\nSaved:",
        OUTPUT_DIR / "integrated_volatility_histogram.png",
    )


    print("\n--- BASELINE ARITHMETIC ASIAN CALL RESULTS ---")

    asian_result, average_ratio = run_lmsv_asian_monte_carlo(
        params=params,
        number_of_steps=252,
        number_of_paths=50000,
        batch_size=5000,
        seed=20260719,
        spot_bump=0.1,
    )

    for key, value in asian_result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")

    print("\n--- ASIAN DELTA: BUMP SENSITIVITY ---")

    for h in [0.05, 0.10, 0.25, 0.50]:
        bump_result = estimate_asian_quantities(
            params=params,
            average_ratio=average_ratio,
            spot_bump=h,
        )

        print(
            f"h={h:.2f} | "
            f"pathwise={bump_result['delta_pathwise']:.8f} | "
            f"finite_difference="
            f"{bump_result['delta_finite_difference']:.8f} | "
            f"difference={bump_result['delta_difference']:.8e}"
        )


if __name__ == "__main__":
    main()