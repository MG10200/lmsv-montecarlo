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
    plt.savefig("integrated_volatility_histogram.png", dpi=200)
    plt.close()

    print("\nSaved: integrated_volatility_histogram.png")


if __name__ == "__main__":
    main()
