# Monte Carlo Implementation Guide

## 1. General structure of the project

The numerical implementation developed for this thesis is organized into two Python files with clearly separated responsibilities.

- **`lmsv_monte_carlo.py`** contains the mathematical and numerical implementation of the Long-Memory Stochastic Volatility (LMSV) model. It defines the model parameters, constructs the covariance matrix of fractional Gaussian noise, performs the Cholesky simulation, generates the fractional Ornstein--Uhlenbeck volatility factor, computes integrated volatility, evaluates European-call prices and Deltas through conditional Black--Scholes valuation, extracts Black--Scholes implied volatility, validates the constant-volatility limit, and contains the simulation and estimation routines for continuously monitored arithmetic Asian calls.

- **`lmsv_experiments.py`** is the driver of the numerical experiments. It imports the reusable routines from `lmsv_monte_carlo.py`, changes the required parameters for each experiment, stores numerical outputs, and generates the CSV files and figures used in the numerical section of the thesis.

This separation keeps the **model and numerical engine** in one file and the **experiments** in a second file. The same simulation engine can therefore be reused across different parameter configurations without rewriting the mathematical implementation.

A key distinction concerns the treatment of European and Asian options.

For European calls, the implementation does **not** simulate complete stock-price paths. Under the assumed independence between the Brownian motion driving the asset and the fractional noise driving volatility, the Brownian asset-price randomness can be integrated out conditionally on the volatility path. The Monte Carlo simulation therefore concentrates on the fractional volatility factor and the resulting future integrated volatility.

The European-call workflow is

```text
Model parameters
        ↓
Time grid
        ↓
Fractional Gaussian-noise covariance matrix Γ
        ↓
Cholesky factor L
        ↓
Correlated fractional Gaussian-noise increments
        ↓
Fractional Ornstein--Uhlenbeck factor Y
        ↓
Bounded logistic volatility σ(Y)
        ↓
Integrated variance V_{0,T}
        ↓
Integrated volatility U_{0,T}
        ↓
Conditional Black--Scholes-type price and Delta
        ↓
Monte Carlo averages
        ↓
LMSV price and Delta
        ↓
Black--Scholes implied volatility
        ↓
Black--Scholes Delta and hedging bias
```

For the continuously monitored arithmetic Asian call, the payoff depends on the asset-price path through

```text
A_T = (1/T) ∫_0^T S_t dt.
```

The conditional reduction used for the European call is therefore not sufficient. The code explicitly simulates the asset dynamics together with the fractional volatility factor and approximates the continuous arithmetic average on the numerical grid.

The Asian-call workflow is

```text
Model parameters
        ↓
Time grid
        ↓
Fractional Gaussian-noise covariance matrix Γ
        ↓
Cholesky factor L
        ↓
Fractional Gaussian-noise increments
        ↓
Fractional Ornstein--Uhlenbeck factor Y
        ↓
Bounded logistic volatility σ(Y)
        ↓
Independent Brownian shocks for the stock
        ↓
Stock-price ratio S_t / S_0
        ↓
Trapezoidal approximation of the arithmetic average
        ↓
Average ratio A_T / S_0
        ↓
Asian payoff
        ↓
Monte Carlo price
        ↓
Pathwise and finite-difference Deltas
```

Only the quantities required by the estimators are retained; complete simulated stock-price and volatility-factor trajectories are not stored.

---

# 2. The file `lmsv_monte_carlo.py`

This file contains the reusable numerical core of the project.

## 2.1 Imported libraries

The file imports

```python
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import brentq
from scipy.special import ndtr
```

The roles are:

- `dataclass`: groups the model parameters into one object;
- `exp`, `log`, `sqrt`: elementary functions used in pricing and simulation calculations;
- `perf_counter`: measures execution time;
- `numpy`: arrays, random-number generation and vectorized numerical calculations;
- `matplotlib`: creation of numerical figures when the core file is executed directly;
- `scipy.linalg.cholesky`: Cholesky factorization of the fractional Gaussian-noise covariance matrix;
- `scipy.optimize.brentq`: numerical inversion of the Black--Scholes price to obtain implied volatility;
- `scipy.special.ndtr`: standard normal cumulative distribution function Φ.

---

## 2.2 `LMSVParameters`: collection of model parameters

The program uses a frozen dataclass, `LMSVParameters`, to store the parameters of a numerical specification in one object. The fields correspond directly to the notation used in the thesis.

| Python variable | Mathematical meaning |
|---|---|
| `spot` | \(S_0\) |
| `strike` | \(K\) |
| `maturity` | \(T\) |
| `rate` | \(r\) |
| `hurst` | \(H\) |
| `mean_reversion` | \(\lambda\) |
| `vol_of_vol` | \(\beta\) |
| `initial_factor` | \(Y_0\) |
| `sigma_min` | \(\sigma_{\min}\) |
| `sigma_max` | \(\sigma_{\max}\) |
| `sigma_slope` | slope parameter \(a\) |

The number of time steps, number of Monte Carlo paths, batch size and random seed are supplied to the simulation functions rather than stored in the dataclass.

Using `frozen=True` prevents accidental changes to a parameter object after creation. When an experiment requires a different value, `lmsv_experiments.py` creates a modified copy using `dataclasses.replace`.

For the baseline specification,

```text
S₀ = 100,   K = 100,   T = 1,   r = 0.02
H = 0.70,   λ = 2,     β = 0.35,   Y₀ = 0
σ_min = 0.15,   σ_max = 0.30,   a = 1.
```

The principal baseline experiments use **252 time steps**, **50,000 Monte Carlo paths**, batches of **5,000 paths**, and the seed **20260719**.

---

## 2.3 Bounded logistic volatility function

The volatility factor \(Y_t\) is Gaussian and can take positive or negative values, so it cannot itself represent volatility. The code transforms it through the bounded logistic specification

```text
σ(y) = σ_min + (σ_max − σ_min) / [1 + exp(−ay)].
```

For numerical evaluation, the implementation uses the algebraically equivalent form

```text
σ(y)
=
σ_min
+
(σ_max − σ_min) · [1 + tanh(ay/2)] / 2.
```

The `tanh` representation avoids evaluating very large exponentials when \(|ay|\) is large.

The resulting volatility remains between the prescribed bounds. The simulation routines also impose

```text
σ_max ≤ 2 σ_min,
```

which is the sufficient condition adopted in the thesis for the required sublinearity property.

The parameter `sigma_slope`, denoted by \(a\), controls the steepness of the transition between the two bounds.

---

## 2.4 Construction of the fractional Gaussian-noise covariance

The simulation uses a uniform time grid

```text
t_k = k Δt,
Δt = T / n,
k = 0, ..., n.
```

The fractional Brownian increments

```text
ΔW_k^H = W^H_{t_{k+1}} − W^H_{t_k}
```

form fractional Gaussian noise (fGn). On the uniform grid, their covariance at lag \(\ell\) is

```text
γ_H(ℓ)
=
(Δt)^(2H) / 2
· [ |ℓ+1|^(2H) − 2|ℓ|^(2H) + |ℓ−1|^(2H) ].
```

The code constructs the covariance matrix

```text
Γ_ij = γ_H(|i−j|).
```

Because the covariance depends only on the lag, the matrix is symmetric Toeplitz. The same matrix applies to all Monte Carlo paths for fixed \(n\), \(T\), and \(H\).

This step introduces the temporal dependence characteristic of fractional Brownian motion. Monte Carlo replications are independent across paths, whereas the fractional increments within each path are correlated.

---

## 2.5 Cholesky decomposition and generation of fGn

Once \(\Gamma\) has been constructed, the code computes

```text
Γ = L Lᵀ
```

using SciPy's Cholesky routine. If

```text
Z ~ N(0, I_n),
```

then

```text
ΔW^H = L Z
```

has covariance \(\Gamma\).

In finite-precision arithmetic, the covariance matrix may be extremely close to singular. The implementation therefore adds a small diagonal regularization,

```text
Γ_ε = Γ + ε I,
```

where

```text
ε = 10^(−13) · max{1, max_i Γ_ii}.
```

The perturbation is introduced solely for numerical robustness. The Cholesky factor is computed once for a given numerical specification and reused across the Monte Carlo paths generated under that specification.

---

## 2.6 Simulation of the fractional Ornstein--Uhlenbeck factor

The volatility factor follows

```text
dY_t = −λY_t dt + β dW_t^H.
```

Its explicit representation is

```text
Y_t
=
exp(−λt)
· [Y₀ + β ∫₀ᵗ exp(λs) dW_s^H].
```

With

```text
q = exp(−λ Δt),
```

the implementation propagates the factor according to

```text
Y_{t_{k+1}} = q Y_{t_k} + β q ΔW_k^H.
```

This recursion corresponds to a left-endpoint approximation of the deterministic integrand appearing in the explicit fOU representation.

At each time point, the current factor is passed through the bounded logistic function to obtain instantaneous volatility.

---

# 3. European call implementation

## 3.1 Black--Scholes helper routines

The core file contains functions for the standard Black--Scholes call price and Delta. For a call,

```text
C^BS = S₀ Φ(d₁) − K e^(−rT) Φ(d₂),
```

with

```text
d₁ = [log(S₀/K) + (r + σ²/2)T] / [σ√T],
d₂ = d₁ − σ√T.
```

The corresponding Delta is

```text
Δ^BS = Φ(d₁).
```

These functions are used to evaluate the benchmark, recover the constant-volatility limit, invert the LMSV price into a Black--Scholes implied volatility, and calculate the Black--Scholes Delta entering the hedging-bias comparison.

The standard normal CDF is evaluated through `scipy.special.ndtr`.

---

## 3.2 Integrated variance and integrated volatility

For the European call, the volatility path enters the conditional pricing formula through

```text
V_{0,T} = ∫₀ᵀ σ²(Y_s) ds
```

and

```text
U_{0,T} = √V_{0,T}.
```

For each simulated factor trajectory \(m\), the code uses the left-point Riemann approximation

```text
V_{0,T}^{(m)}
≈
Σ_{k=0}^{n−1} σ²(Y_{t_k}^{(m)}) Δt,
```

and then computes

```text
U_{0,T}^{(m)} = √V_{0,T}^{(m)}.
```

The function `simulate_integrated_volatility` returns the resulting sample of integrated volatilities.

---

## 3.3 Batch processing for the European simulation

The Monte Carlo sample can contain tens of thousands of paths. The code therefore processes paths in batches.

For example, with \(M=50{,}000\) and a batch size of \(5{,}000\), each batch:

1. generates the required fGn paths;
2. propagates the corresponding fOU factors;
3. accumulates integrated variance;
4. computes the integrated volatilities;
5. discards the complete factor trajectories.

Only the \(M\) integrated-volatility values required for pricing and Delta estimation are retained.

Batching reduces memory requirements without changing the Monte Carlo estimator.

---

## 3.4 Conditional price and Delta

At time zero, define

```text
m₀ = log[S₀ / (K e^(−rT))].
```

For each simulated integrated volatility \(U_{0,T}^{(m)}\),

```text
d₁^(m) = m₀ / U_{0,T}^{(m)} + U_{0,T}^{(m)} / 2,
d₂^(m) = d₁^(m) − U_{0,T}^{(m)}.
```

The conditional call price is

```text
C^(m)
=
S₀ Φ(d₁^(m))
−
K e^(−rT) Φ(d₂^(m)),
```

and the conditional LMSV Delta contribution is

```text
Δ^(m) = Φ(d₁^(m)).
```

This reduction is possible because the Brownian motion driving the stock is independent of the fractional noise driving volatility. Conditional on the volatility path, the remaining stock-price Brownian randomness can be integrated out analytically.

---

## 3.5 European Monte Carlo estimators and standard errors

The LMSV call price and Delta are estimated as

```text
Ĉ₀^LMSV = (1/M) Σ C^(m),
Δ̂₀^LMSV = (1/M) Σ Δ^(m).
```

The simulation therefore averages conditional Black--Scholes-type quantities rather than terminal simulated European option payoffs.

For a generic sample \(X^{(1)},\ldots,X^{(M)}\), the estimated standard error of its sample mean is

```text
SE_hat = s_X / √M,
```

where \(s_X\) is the sample standard deviation with one degree-of-freedom correction.

Approximate 95% Monte Carlo confidence intervals are formed as

```text
sample mean ± 1.96 · SE_hat.
```

---

## 3.6 Black--Scholes implied volatility

After estimating the LMSV European-call price, the code determines the Black--Scholes implied volatility \(\sigma^i\) by solving

```text
C^BS(S₀, K, T, r, σ^i) = Ĉ₀^LMSV.
```

The root is found numerically with Brent's method (`brentq`) over

```text
[10^(−10), 5].
```

Before inversion, the code checks the call no-arbitrage bounds

```text
max{S₀ − K e^(−rT), 0}
≤ Ĉ₀^LMSV ≤ S₀.
```

---

## 3.7 Black--Scholes Delta and hedging bias

Once \(\sigma^i\) has been obtained, the Black--Scholes Delta is evaluated at that implied volatility.

The hedging bias is defined as

```text
Bias₀ = Δ₀^BS(σ^i) − Δ̂₀^LMSV.
```

Hence:

- a negative bias means that the Black--Scholes implied-volatility Delta is below the LMSV Delta;
- a positive bias means that it is above the LMSV Delta.

This quantity measures the initial hedge-ratio discrepancy. It is not the terminal profit or loss of a dynamically rebalanced hedging strategy.

---

## 3.8 Black--Scholes-limit validation

The function `validate_black_scholes_limit` sets

```text
β = 0.
```

With \(Y_0=0\), the factor remains zero. Under the baseline logistic specification,

```text
σ(0) = (σ_min + σ_max)/2 = 0.225.
```

The simulated integrated volatility is therefore deterministic and the LMSV price and Delta coincide with their Black--Scholes counterparts to displayed numerical precision.

This checks the integrated-volatility construction and the conditional pricing layer in the constant-volatility limit.

---

# 4. Arithmetic Asian call implementation

## 4.1 Why stock-price simulation is required

The extension considers a continuously monitored arithmetic Asian call with payoff

```text
(A_T − K)^+,
```

where

```text
A_T = (1/T) ∫₀ᵀ S_t dt.
```

Unlike the European payoff, this payoff depends on the trajectory of the stock over the entire interval. The European conditional-mixture reduction to the scalar \(U_{0,T}\) is therefore insufficient for evaluating the Asian payoff.

The function `simulate_asian_average_ratios` explicitly propagates the stock dynamics on the same numerical grid used for the volatility factor.

---

## 4.2 Independent random streams for volatility and stock noise

The LMSV specification assumes independence between the fractional noise driving volatility and the Brownian motion driving the stock.

The Asian simulation makes this assumption explicit by creating a `SeedSequence` from the user-supplied seed and spawning two pseudo-random streams:

```python
seed_sequence = np.random.SeedSequence(seed)
fgn_seed, stock_seed = seed_sequence.spawn(2)
```

One generator produces the independent normals transformed into fGn increments; the other produces the standard normal Brownian shocks used in the stock update.

This preserves reproducibility while keeping the two simulated noise sources separate.

---

## 4.3 Simulation of the stock-price ratio

Rather than propagating \(S_t\) directly, the implementation works with

```text
R_t = S_t / S₀.
```

The stock dynamics are multiplicative, so \(R_t\) does not depend on the numerical value of the initial spot. The ratio starts from

```text
R_0 = 1.
```

Over one time step, the code uses

```text
R_{t_{k+1}}
=
R_{t_k}
exp[
    (r − σ²(Y_{t_k})/2) Δt
    + σ(Y_{t_k}) √Δt Z_k
],
```

where the \(Z_k\) are independent standard normal shocks from the stock-noise random stream.

This is the exact geometric-Brownian update over each interval conditional on the volatility value held fixed at its left endpoint.

---

## 4.4 Trapezoidal approximation of the continuous arithmetic average

The continuous average can be written as

```text
A_T / S₀
=
(1/T) ∫₀ᵀ (S_t/S₀) dt.
```

The implementation approximates the integral through the trapezoidal rule:

```text
A_T / S₀
≈
(1/T)
Σ_{k=0}^{n−1}
0.5 (R_{t_k} + R_{t_{k+1}}) Δt.
```

The simulation retains only this **average ratio** for each Monte Carlo path.

Consequently,

```text
A_T ≈ S₀ · average_ratio.
```

Complete stock-price and volatility-factor trajectories do not need to be stored after their contribution to the running average has been accumulated.

---

## 4.5 Batch processing for the Asian simulation

The Asian simulation is also performed in batches. For each batch, the algorithm:

1. generates the fGn increments;
2. initializes the fOU factor and stock ratio;
3. generates stock Brownian shocks step by step;
4. updates the stock ratio;
5. accumulates the trapezoidal integral;
6. updates the fOU factor;
7. stores only the final average ratio for each path.

Thus memory use remains controlled even though both volatility and stock dynamics must be simulated.

---

## 4.6 Asian option price estimator

For each simulated average ratio \(R_A^{(m)}\),

```text
A_T^(m) = S₀ R_A^(m).
```

The discounted payoff sample is

```text
e^(−rT) max(A_T^(m) − K, 0).
```

The Monte Carlo price estimator is

```text
Ĉ₀^Asian
=
(1/M)
Σ_{m=1}^M
e^(−rT)
max(A_T^(m) − K, 0).
```

The code also stores the mean undiscounted payoff as a diagnostic quantity and reports the sample mean and standard deviation of the simulated arithmetic average.

The price standard error is computed from the discounted payoff samples in the usual way.

---

## 4.7 Pathwise Asian Delta

Because

```text
A_T = S₀ R_A
```

for a fixed simulated ratio \(R_A\),

```text
∂A_T / ∂S₀ = R_A.
```

Differentiating the discounted payoff pathwise gives the sample contribution

```text
e^(−rT)
1_{A_T > K}
R_A.
```

The pathwise Delta estimator implemented in the code is therefore

```text
Δ̂_PW
=
(1/M)
Σ_{m=1}^M
e^(−rT)
1_{A_T^(m) > K}
R_A^(m).
```

Its Monte Carlo standard error is computed directly from these pathwise Delta samples.

---

## 4.8 Centered finite-difference Asian Delta

The code provides an independent numerical check of the pathwise estimator through a centered finite difference. For a positive spot bump \(h<S_0\),

```text
Δ̂_FD
=
[C(S₀+h) − C(S₀−h)] / (2h).
```

The implementation does not rerun the Monte Carlo simulation for the bumped spots. Since the same simulated average ratio can be rescaled by either \(S_0+h\) or \(S_0-h\), it evaluates

```text
payoff_plus
=
max[(S₀+h) R_A − K, 0],

payoff_minus
=
max[(S₀−h) R_A − K, 0].
```

The finite-difference sample is then

```text
e^(−rT)
[payoff_plus − payoff_minus] / (2h).
```

The same simulated paths are therefore used on both sides of the finite difference. This is a common-random-numbers variance-reduction technique and makes the comparison between the two bumped prices substantially less noisy than independent simulations would be.

---

## 4.9 Difference between the two Asian Delta estimators

The pathwise and finite-difference estimators are computed from the same Monte Carlo paths and are therefore dependent.

The code consequently evaluates their difference path by path:

```text
D^(m)
=
Δ_FD^(m) − Δ_PW^(m),
```

and reports

```text
mean(D)
```

together with

```text
SE(mean(D)).
```

It does **not** combine the two individual standard errors as if the estimators were independent.

This provides a direct numerical diagnostic of agreement between the pathwise and finite-difference methods.

---

## 4.10 `run_lmsv_asian_monte_carlo`

The function `run_lmsv_asian_monte_carlo` combines the two stages of the Asian computation:

1. `simulate_asian_average_ratios` generates the Monte Carlo sample of \(A_T/S_0\);
2. `estimate_asian_quantities` evaluates the price and both Delta estimators from that sample.

The returned result also records elapsed time, number of steps and number of paths.

---

# 5. The file `lmsv_experiments.py`

The second file does not redefine the LMSV model. It imports the reusable functions from `lmsv_monte_carlo.py` and organizes the experiments reported in the thesis.

The current imports from the core module are

```python
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
```

It also uses `csv`, `dataclasses.replace`, `pathlib.Path`, NumPy and Matplotlib.

---

## 5.1 Output directory

The experiment script defines

```python
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
```

and writes the CSV files and figures generated by the experiment functions to this directory.

The exact location is therefore relative to the working directory from which `lmsv_experiments.py` is executed.

---

## 5.2 Baseline specification

A common `LMSVParameters()` object provides the baseline model specification.

Whenever an experiment changes one parameter, such as \(H\) or \(K\), the remaining parameters are held fixed. `dataclasses.replace` is used to create the modified parameter object without mutating the baseline specification.

---

## 5.3 Black--Scholes-limit check

The first experiment calls `validate_black_scholes_limit`.

It switches off stochastic volatility by setting \(\beta=0\) and compares the resulting LMSV price and Delta with the corresponding Black--Scholes quantities at the constant volatility

```text
σ(0) = 0.225.
```

The script also prints the Black--Scholes benchmark price and Delta under the baseline spot, strike, maturity and interest rate.

---

## 5.4 Monte Carlo convergence experiment

The convergence study fixes

```text
n = 252
```

and runs the European simulation for

```text
M ∈ {1,000, 5,000, 10,000, 25,000, 50,000}.
```

For each \(M\), the script stores price and Delta estimates, standard errors, approximate 95% confidence intervals and execution time.

The convergence figure compares the observed European-call price standard error with an \(M^{-1/2}\) reference decay.

The corresponding numerical output is stored in

```text
monte_carlo_convergence.csv
```

and the figure in

```text
monte_carlo_convergence.png.
```

---

## 5.5 European time-grid stability experiment

This experiment fixes

```text
M = 25,000
```

and varies

```text
n ∈ {32, 64, 128, 252, 512}.
```

For each grid, the fGn covariance matrix and Cholesky factor are rebuilt because \(\Delta t=T/n\) changes.

The script records the European-call price and Delta, their standard errors, the mean and standard deviation of integrated volatility, and execution time.

The price figure uses approximate 95% Monte Carlo error bars.

Outputs:

```text
time_grid_stability.csv
time_grid_stability.png
```

---

## 5.6 Strike and hedging-bias experiment

The strike study first generates a single sample of integrated volatilities and then reuses that sample for

```text
K ∈ {80, 90, 95, 100, 105, 110, 120}.
```

This common Monte Carlo sample improves comparability across strikes.

For each strike, the script records:

- adjusted log-moneyness;
- LMSV price and price standard error;
- LMSV Delta and Delta standard error;
- Black--Scholes implied volatility;
- Black--Scholes implied-volatility Delta;
- hedging bias.

It also computes

```text
F_{0,T} = S₀ e^(rT)
```

to identify the at-the-money-forward strike shown in the figure.

Outputs:

```text
strike_hedging_bias.csv
strike_hedging_bias.png
```

---

## 5.7 European Hurst-parameter sensitivity

The European Hurst experiment uses

```text
H ∈ {0.55, 0.65, 0.75, 0.85, 0.95},
M = 25,000,
n = 252.
```

For every \(H\), a new fGn covariance matrix and Cholesky factor are required.

The stored quantities include:

- price and price standard error;
- Delta and Delta standard error;
- implied volatility;
- hedging bias;
- mean integrated volatility;
- standard deviation of integrated volatility.

The generated figure focuses on the standard deviation of \(U_{0,T}\).

Outputs:

```text
hurst_sensitivity.csv
hurst_sensitivity.png
```

The rows generated by this experiment are also reused later in the European--Asian Hurst comparison.

---

# 6. Arithmetic Asian experiments

## 6.1 Asian baseline experiment

`asian_baseline_study` evaluates the continuously monitored arithmetic Asian call under the baseline LMSV specification using

```text
n = 252,
M = 50,000,
batch size = 5,000,
seed = 20260719,
spot bump h = 0.10.
```

The continuous arithmetic average is approximated by the trapezoidal rule.

The stored output includes:

- Asian call price and standard error;
- mean undiscounted payoff;
- pathwise Delta and standard error;
- finite-difference Delta and standard error;
- difference between the two Delta estimators and its standard error;
- spot bump;
- mean and standard deviation of the simulated arithmetic average;
- execution time.

Output:

```text
asian_baseline.csv
```

The function also returns the simulated average-ratio sample so that it can be reused in the bump-sensitivity experiment.

---

## 6.2 Asian Delta bump-sensitivity experiment

`asian_delta_bump_study` compares the pathwise Delta with centered finite differences for

```text
h ∈ {0.05, 0.10, 0.25, 0.50}.
```

The same baseline sample of average ratios is reused for every bump. Therefore, changes in the finite-difference estimate across \(h\) are not confounded by independent Monte Carlo samples.

For each bump, the script stores:

- pathwise Delta and standard error;
- finite-difference Delta and standard error;
- their difference;
- standard error of the pathwise difference;
- the corresponding difference z-score.

Outputs:

```text
asian_delta_bump_sensitivity.csv
asian_delta_bump_sensitivity.png
```

The figure displays the pathwise Delta as a horizontal reference and the finite-difference estimate as a function of the spot perturbation.

---

## 6.3 Asian time-grid stability

`asian_time_grid_stability_study` studies

```text
n ∈ {32, 64, 128, 252, 512}
```

with

```text
M = 25,000.
```

For every grid, the complete Asian simulation is rerun because both the fractional volatility discretization and the numerical approximation of the continuous arithmetic average depend on the time grid.

The script records:

- Asian price and standard error;
- pathwise Delta and standard error;
- mean and standard deviation of the simulated arithmetic average;
- execution time.

The generated price figure uses approximate 95% Monte Carlo error bars.

Outputs:

```text
asian_time_grid_stability.csv
asian_time_grid_stability.png
```

---

## 6.4 European--Asian comparison across the Hurst parameter

`european_asian_hurst_comparison_study` compares European and arithmetic Asian calls at the same Hurst values used in the European sensitivity experiment.

The European rows are reused rather than recomputed. For each \(H\), the Asian simulation is then run with the same baseline model parameters, number of time steps, number of paths and seed.

The stored quantities include:

- European and Asian prices and their standard errors;
- European-minus-Asian price difference;
- European and Asian Deltas and their standard errors;
- European-minus-Asian Delta difference;
- mean and standard deviation of the Asian arithmetic average;
- price and Delta differences scaled by the corresponding Asian standard errors.

Output:

```text
european_asian_hurst_comparison.csv
```

Two figures are generated:

```text
european_asian_price_hurst.png
european_asian_delta_hurst.png
```

They compare European and Asian prices and Deltas, respectively, across the selected values of \(H\).

---

# 7. How the two files work together

The complete organization can be summarized as follows.

```text
                       lmsv_experiments.py
                                │
                                │ selects experiment and parameters
                                ▼
                         LMSVParameters
                                │
                                ▼
                       lmsv_monte_carlo.py
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
        European-call branch              Asian-call branch
                │                                │
        build fGn covariance               build fGn covariance
        Cholesky factorization             Cholesky factorization
        simulate fGn                       simulate fGn
        propagate fOU                      propagate fOU
        compute σ(Y)                       compute σ(Y)
        accumulate V_{0,T}                 simulate stock shocks
        compute U_{0,T}                    propagate S_t/S_0
                │                          accumulate average ratio
                ▼                                │
        conditional price/Delta                  ▼
        implied volatility                 Asian payoff
        BS Delta and bias                  pathwise Delta
                │                          finite-difference Delta
                └───────────────┬────────────────┘
                                ▼
                        numerical results
                                │
                                ▼
                     CSV files and PNG figures
```

The core file answers:

> **How is a given LMSV numerical specification evaluated?**

The experiment file answers:

> **Which specifications and diagnostics are evaluated to produce the numerical analysis of the thesis?**

---

# 8. Main numerical choices to remember

The most important implementation choices are:

- **Fractional Gaussian noise rather than reconstructed fBM paths:** the fOU recursion uses the increments directly.
- **Cholesky simulation:** imposes the required finite-dimensional Gaussian covariance structure transparently.
- **Small diagonal regularization:** improves numerical robustness of the Cholesky factorization.
- **Bounded logistic volatility:** keeps volatility positive and within the prescribed bounds.
- **`tanh` implementation of the logistic function:** avoids numerical overflow from large exponentials.
- **Left-point treatment of volatility:** used in the fOU update, integrated-variance Riemann sum, and stock update over each interval.
- **Batch processing:** controls memory requirements without changing the estimators.
- **Conditional European pricing:** exploits independence of the stock and volatility noises and reduces the European problem to the simulated distribution of \(U_{0,T}\).
- **Explicit Asian stock simulation:** required because the arithmetic Asian payoff depends on the asset-price path.
- **Stock-ratio representation:** simulates \(S_t/S_0\), making the same simulated ratio reusable for spot perturbations.
- **Trapezoidal integration for the Asian average:** approximates the continuously monitored arithmetic average on the numerical grid.
- **Independent pseudo-random streams:** make the independence of fractional volatility noise and stock Brownian noise explicit in the Asian simulation.
- **Pathwise Asian Delta:** exploits the multiplicative dependence \(A_T=S_0R_A\).
- **Centered finite differences with common random numbers:** provide a low-noise numerical check of the pathwise Delta.
- **Pathwise analysis of the Delta-estimator difference:** correctly accounts for dependence between the two estimators.
- **Brent's method:** provides robust inversion of the European LMSV price into Black--Scholes implied volatility.
- **Common integrated-volatility sample across strikes:** improves comparability in the hedging-bias experiment.
- **Fixed seeds:** make the reported numerical experiments reproducible.

---

# 9. Reproducing the numerical analysis

The external Python dependencies are

```text
numpy
scipy
matplotlib
```

and are listed in `requirements.txt`.

From the repository root, the core implementation can be run with

```bash
python src/lmsv_monte_carlo.py
```

and the complete collection of experiments with

```bash
python src/lmsv_experiments.py
```

When the core file is executed directly, it performs the Black--Scholes-limit check, reports the baseline European LMSV results, saves the integrated-volatility histogram, reports the baseline Asian results, and prints an Asian Delta bump-sensitivity comparison.

When `lmsv_experiments.py` is executed, it runs the full sequence of European and Asian numerical experiments described above and writes its CSV and PNG outputs to the `results` directory relative to the current working directory.

---

# 10. Summary

The implementation contains two complementary Monte Carlo branches.

For **European calls**, it simulates the fractional volatility dynamics, reduces each volatility path to future integrated volatility, evaluates conditional Black--Scholes-type prices and Deltas, and compares the LMSV Delta with the Black--Scholes implied-volatility Delta.

For **continuously monitored arithmetic Asian calls**, it additionally simulates the stock dynamics, approximates the continuous arithmetic average by the trapezoidal rule, estimates the option price from discounted payoffs, and computes the initial Delta using both pathwise differentiation and centered finite differences with common random numbers.

The experiment script then uses these reusable numerical routines to perform validation, convergence and grid-stability checks, study European hedging bias and Hurst sensitivity, validate the Asian Delta estimator, and compare European and Asian option behavior under the same LMSV specifications.
