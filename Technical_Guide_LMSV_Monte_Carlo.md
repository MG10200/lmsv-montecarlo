# Technical Guide to the Monte Carlo Implementation

## 1. General structure of the project

The numerical implementation developed for this thesis is organized into two Python files with clearly separated responsibilities.

- **`lmsv_monte_carlo.py`** contains the mathematical and numerical implementation of the Long-Memory Stochastic Volatility (LMSV) model. It defines the model parameters, constructs the covariance matrix of fractional Gaussian noise, performs the Cholesky simulation, generates the fractional Ornstein--Uhlenbeck volatility factor, computes the integrated volatility, evaluates LMSV option prices and Deltas, extracts the Black--Scholes implied volatility, and validates the implementation in the constant-volatility limit.

- **`lmsv_experiments.py`** is the driver of the numerical experiments. It imports the routines from `lmsv_monte_carlo.py`, changes the required parameters for each experiment, stores numerical outputs, and generates the CSV files and figures used in the numerical section of the thesis.

This separation keeps the **model and numerical engine** in one file and the **experiments** in a second file. Therefore, the same simulation engine can be reused without rewriting the mathematical implementation every time a parameter is changed.

A crucial point is that the current European-call implementation does **not** simulate complete stock-price paths. Under the assumed independence between the Brownian motion driving the asset and the fractional noise driving volatility, the Brownian asset-price randomness can be integrated out conditionally on the volatility path. The Monte Carlo simulation therefore concentrates on the fractional volatility factor and on the random integrated volatility.

The main workflow is

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
- `exp`, `log`, `sqrt`: elementary functions used in Black--Scholes calculations;
- `perf_counter`: measures execution time;
- `numpy`: arrays, random-number generation and vectorized numerical calculations;
- `matplotlib`: creation of numerical figures;
- `scipy.linalg.cholesky`: Cholesky factorization of the fractional Gaussian-noise covariance matrix;
- `scipy.optimize.brentq`: numerical inversion of the Black--Scholes price to obtain implied volatility;
- `scipy.special.ndtr`: standard normal cumulative distribution function Φ.

---

## 2.2 `LMSVParameters`: collection of model parameters

The program uses a frozen dataclass, `LMSVParameters`, to store the parameters of a numerical specification in one object. The main fields correspond directly to the notation used in the thesis.

| Python variable | Mathematical meaning |
|---|---|
| `spot` | S₀ |
| `strike` | K |
| `maturity` | T |
| `rate` | r |
| `hurst` | H |
| `mean_reversion` | λ |
| `vol_of_vol` | β |
| `initial_factor` | Y₀ |
| `sigma_min` | σ_min |
| `sigma_max` | σ_max |
| `sigma_slope` | slope parameter a |
| number of steps | n |
| number of paths | M |
| batch size | number of paths processed simultaneously |
| random seed | seed used for reproducibility |

Using `frozen=True` prevents accidental changes to the parameter object after creation. When an experiment needs a different value, the second script creates a modified copy rather than mutating the original specification.

For the baseline experiment the numerical values used in the thesis are

```text
S₀ = 100,   K = 100,   T = 1,   r = 0.02
H = 0.70,   λ = 2,     β = 0.35,   Y₀ = 0
σ_min = 0.15,   σ_max = 0.30,   a = 1
```

The baseline discretization uses **252 time steps**, **50,000 Monte Carlo paths** and batches of **5,000 paths**.

---

## 2.3 Black--Scholes helper routines

The file contains functions for the standard Black--Scholes call price and Delta.

For a call,

```text
C^BS = S₀ Φ(d₁) − K e^(−rT) Φ(d₂)
```

with

```text
d₁ = [ log(S₀/K) + (r + σ²/2)T ] / [ σ√T ]

d₂ = d₁ − σ√T
```

The corresponding Delta is

```text
Δ^BS = Φ(d₁)
```

These routines are not the main pricing engine of the LMSV simulation. They serve three purposes:

1. to evaluate the Black--Scholes benchmark;
2. to recover the constant-volatility limit;
3. to compute the Black--Scholes implied volatility and the corresponding Black--Scholes Delta used in the hedging-bias comparison.

The normal CDF is evaluated through `ndtr`, which is a vectorized and numerically reliable SciPy implementation of Φ.

---

## 2.4 Bounded logistic volatility function

The volatility factor Y_t is Gaussian and can take positive or negative values, so it cannot itself represent volatility. The code therefore transforms it through the bounded logistic specification

```text
σ(y) = σ_min + (σ_max − σ_min) / [1 + exp(−ay)]
```

For numerical evaluation the implementation uses the algebraically equivalent form

```text
σ(y) = σ_min
       + (σ_max − σ_min) · [1 + tanh(ay/2)] / 2
```

The `tanh` representation is preferable numerically because it avoids evaluating extremely large exponentials when |ay| is large.

The function always remains between the two volatility bounds:

```text
σ_min < σ(y) < σ_max
```

The code also imposes

```text
σ_max ≤ 2 σ_min
```

which is the sufficient condition used in the thesis to guarantee the required sublinearity property.

The parameter `sigma_slope`, denoted by a, controls only the steepness of the transition between the two volatility bounds.

---

## 2.5 Construction of the fractional Gaussian-noise covariance

The simulation uses a uniform time grid

```text
t_k = k Δt

Δt = T / n

k = 0, ..., n
```

The fractional Brownian increments are

```text
ΔW_k^H = W^H_(t_{k+1}) − W^H_(t_k)
```

On an equally spaced grid they form fractional Gaussian noise (fGn). Their covariance at lag ℓ is

```text
γ_H(ℓ)
=
(Δt)^(2H) / 2
· [ |ℓ+1|^(2H) − 2|ℓ|^(2H) + |ℓ−1|^(2H) ]
```

The code evaluates this function for the lags required to construct

```text
Γ_ij = γ_H(|i−j|)
```

Because the covariance depends only on |i−j|, the matrix is symmetric Toeplitz. The same covariance matrix applies to every simulated path as long as n, T and H remain fixed.

This step is what inserts the temporal dependence characteristic of fractional Brownian motion into the simulation. The paths are independent **across Monte Carlo replications**, but the increments **within each path** are correlated.

---

## 2.6 Cholesky decomposition and generation of fGn

Once Γ has been constructed, the code computes

```text
Γ = L Lᵀ
```

using SciPy's Cholesky routine.

If

```text
Z ~ N(0, I_n)
```

then

```text
ΔW^H = L Z
```

has covariance

```text
Cov(ΔW^H) = Γ
```

Thus the algorithm converts independent standard normal random variables into Gaussian increments having exactly the covariance structure required by the chosen Hurst parameter.

In finite-precision arithmetic the covariance matrix can be extremely close to singular. The implementation therefore adds a very small diagonal regularization,

```text
Γ_ε = Γ + ε I
```

with

```text
ε = 10^(−13) · max{1, max_i Γ_ii}
```

This perturbation is not intended to change the model. Its purpose is solely to make the factorization numerically robust.

The Cholesky factor is computed once for a given pair (n, H) and then reused for all Monte Carlo paths generated under that specification.

---

## 2.7 Simulation of the fractional Ornstein--Uhlenbeck factor

The volatility factor follows

```text
dY_t = −λY_t dt + β dW_t^H
```

Its explicit representation is

```text
Y_t
=
exp(−λt)
· [ Y₀ + β ∫₀ᵗ exp(λs) dW_s^H ]
```

The implementation introduces

```text
q = exp(−λ Δt)
```

and updates the factor according to

```text
Y_(t_{k+1}) = q Y_(t_k) + β q ΔW_k^H
```

This recursion is obtained by approximating the deterministic integrand in the explicit fOU representation on each interval by its left-endpoint value.

The interpretation is:

- `q Y_(t_k)` produces mean reversion;
- β determines the intensity of factor fluctuations;
- `ΔW_k^H` introduces persistent, temporally dependent fractional shocks.

At every time point the simulated factor is passed through the logistic function to obtain the instantaneous volatility.

---

## 2.8 Integrated variance and integrated volatility

The European-call pricing formula used in the thesis depends on the future volatility path only through the integrated variance

```text
V_{0,T} = ∫₀ᵀ σ²(Y_s) ds
```

and the corresponding integrated volatility

```text
U_{0,T} = √V_{0,T}
```

For every simulated factor trajectory m, the code approximates the integral through the left-point Riemann sum

```text
V_{0,T}^(m)
≈
Σ_(k=0)^(n−1) σ²(Y_(t_k)^(m)) Δt
```

and then computes

```text
U_{0,T}^(m) = √V_{0,T}^(m)
```

This is a central design choice of the implementation. The algorithm does not need to retain the entire factor path once this scalar quantity has been obtained.

---

## 2.9 Batch processing

The total Monte Carlo sample can contain tens of thousands of paths. Simulating all factor trajectories simultaneously would unnecessarily increase memory use.

The code therefore processes the paths in **batches**.

For example, with M = 50,000 and a batch size of 5,000:

1. generate 5,000 fGn paths;
2. propagate the 5,000 corresponding fOU trajectories;
3. accumulate their integrated variances;
4. compute the 5,000 values of U_{0,T};
5. discard the complete trajectories;
6. proceed to the next 5,000 paths.

Only the integrated-volatility values needed for pricing and Delta calculations are retained.

Batching changes memory requirements but does not change the Monte Carlo estimator.

---

## 2.10 Conditional price and Delta for each simulated volatility path

At time 0, define the adjusted log-moneyness

```text
m₀ = log[ S₀ / (K e^(−rT)) ]
```

For each simulated integrated volatility U_{0,T}^(m), the code computes

```text
d₁^(m) = m₀ / U_{0,T}^(m) + U_{0,T}^(m) / 2

d₂^(m) = d₁^(m) − U_{0,T}^(m)
```

The conditional call price is

```text
C^(m)
=
S₀ Φ(d₁^(m))
−
K e^(−rT) Φ(d₂^(m))
```

while the conditional LMSV Delta contribution is

```text
Δ^(m) = Φ(d₁^(m))
```

The reason this reduction is possible is the independence assumption between the Brownian motion driving the stock price and the fractional noise driving volatility. Once the volatility path is fixed, the remaining Brownian randomness in the stock price can be integrated out analytically.

This is why the present European-call simulation is computationally more efficient than a brute-force simulation of S_T.

---

## 2.11 Monte Carlo estimators

The LMSV call price is estimated as

```text
Ĉ₀^LMSV = (1/M) Σ_(m=1)^M C^(m)
```

The LMSV Delta is estimated as

```text
Δ̂₀^LMSV = (1/M) Σ_(m=1)^M Δ^(m)
```

Thus the simulation averages **conditional Black--Scholes-type quantities**, rather than terminal simulated option payoffs.

This is the numerical counterpart of the conditional-mixture representation developed in the theoretical part of the thesis.

---

## 2.12 Monte Carlo standard errors

For a generic simulated quantity X^(1), ..., X^(M), the code evaluates the standard error of the sample mean as

```text
SE_hat = s_X / √M
```

where `s_X` is the sample standard deviation.

An approximate 95% confidence interval is

```text
sample mean ± 1.96 · SE_hat
```

The same logic is applied to both the price estimator and the Delta estimator.

These statistics are used particularly in the Monte Carlo convergence and time-grid studies.

---

## 2.13 Black--Scholes implied volatility

After obtaining the LMSV Monte Carlo price, the code determines the Black--Scholes implied volatility σ^i by solving

```text
C^BS(S₀, K, T, r, σ^i) = Ĉ₀^LMSV
```

There is no analytical formula that inverts the Black--Scholes call price with respect to σ, so the root must be found numerically.

The code uses `brentq` on

```text
[10^(−10), 5]
```

Before performing the inversion, it checks the call no-arbitrage bounds

```text
max{S₀ − K e^(−rT), 0}
≤
Ĉ₀^LMSV
≤
S₀
```

If these inequalities were violated, no admissible positive Black--Scholes volatility could reproduce the estimated price.

---

## 2.14 Black--Scholes Delta and hedging bias

Once σ^i has been obtained, the code evaluates the Black--Scholes Delta at that implied volatility.

The hedging bias is

```text
Bias₀ = Δ₀^BS(σ^i) − Δ̂₀^LMSV
```

Therefore:

- **negative bias:** the Black--Scholes implied-volatility Delta is lower than the LMSV Delta;
- **positive bias:** the Black--Scholes implied-volatility Delta is higher than the LMSV Delta.

This is an **initial hedge-ratio discrepancy**, not the terminal profit or loss of a dynamically rebalanced hedging portfolio.

---

## 2.15 Black--Scholes-limit validation

The implementation includes a specific validation routine in which

```text
β = 0
```

With Y₀ = 0, the volatility factor remains identically zero:

```text
Y_t = 0
```

The logistic volatility is therefore constant:

```text
σ(Y_t) = 0.225
```

and

```text
U_{0,T} = 0.225 √T
```

The LMSV price and Delta produced by the program coincide with their Black--Scholes counterparts to displayed numerical precision.

This verifies that both the integrated-volatility construction and the conditional pricing layer recover the classical model when stochastic volatility is switched off.

---

# 3. The file `lmsv_experiments.py`

The second file does not redefine the model. It imports the reusable routines

```python
from lmsv_monte_carlo import (
    LMSVParameters,
    black_scholes_call,
    black_scholes_delta,
    estimate_lmsv_quantities,
    run_lmsv_monte_carlo,
    validate_black_scholes_limit,
)
```

and uses them to run the experiments reported in the thesis.

It also imports

```python
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
```

`replace()` creates a modified copy of the baseline parameter object while leaving all unspecified parameters unchanged.

---

## 3.1 Output directory

The code defines an output directory using `Path` and creates it automatically if it does not already exist.

This keeps the numerical outputs separate from the source code. The script writes CSV files and PNG figures to this location.

---

## 3.2 Baseline specification

A common baseline parameter set is used as the reference point for the experiments.

Whenever an experiment changes only one quantity, such as H, K, n or M, the remaining model parameters are kept fixed at their baseline values. This makes the comparisons interpretable as controlled sensitivity analyses.

---

## 3.3 Monte Carlo convergence experiment

The convergence study fixes

```text
n = 252
```

and runs the simulation for

```text
M ∈ {1,000, 5,000, 10,000, 25,000, 50,000}
```

For each M, the script stores the estimated call price and its standard error.

The figure compares the observed standard errors with the theoretical reference decay

```text
M^(−1/2)
```

The purpose is not to show monotonic convergence of the estimated prices. Individual Monte Carlo estimates can fluctuate. The relevant check is that statistical uncertainty decreases approximately at the classical square-root rate.

---

## 3.4 Time-grid stability experiment

This experiment fixes

```text
M = 25,000
```

and varies the number of time steps:

```text
n ∈ {32, 64, 128, 252, 512}
```

Every change in n requires rebuilding the fGn covariance matrix because

```text
Δt = T / n
```

changes.

The script records the estimated price, price standard error and mean integrated volatility, and generates a figure with approximate 95% Monte Carlo error bars.

The purpose is to check whether the baseline discretization n = 252 produces stable results relative to finer and coarser grids.

---

## 3.5 Strike and hedging-bias experiment

For the strike study, the script generates **one** sample of integrated volatilities and reuses it for

```text
K ∈ {80, 90, 95, 100, 105, 110, 120}
```

Reusing the same Monte Carlo sample reduces noise in the comparison across strikes because changes in the outputs are not contaminated by independent simulation realizations.

For each strike the code computes:

1. adjusted log-moneyness;
2. LMSV call price;
3. Black--Scholes implied volatility;
4. LMSV Delta;
5. Black--Scholes Delta at implied volatility;
6. hedging bias.

The script also computes the forward price

```text
F_{0,T} = S₀ e^(rT)
```

which identifies the at-the-money-forward strike.

The resulting figure shows the sign change of the hedging bias around this forward strike.

---

## 3.6 Hurst-parameter sensitivity experiment

The Hurst experiment fixes

```text
M = 25,000
n = 252
```

and considers

```text
H ∈ {0.55, 0.65, 0.75, 0.85, 0.95}
```

For every H, the code generates a new covariance matrix and a new Cholesky factor because the temporal covariance of fractional Gaussian noise changes directly with H.

The experiment stores quantities including:

- call price;
- mean integrated volatility;
- standard deviation of integrated volatility.

The corresponding figure focuses on the standard deviation of U_{0,T}. In the selected baseline specification, this dispersion changes only modestly with H, which is one of the numerical observations discussed in the thesis.

---

## 3.7 CSV output

The `csv` module stores numerical results in machine-readable form.

The purpose is twofold:

- preserve the exact numerical values behind the tables and plots;
- make the results reproducible without manually copying numbers from figures.

---

## 3.8 Figure generation

`matplotlib` generates the figures included in the numerical section, including:

- integrated-volatility histogram;
- Monte Carlo convergence plot;
- time-grid stability plot;
- hedging bias across strikes;
- Hurst-parameter sensitivity plot.

The plotting code belongs in the experiment file because figure generation is an output task rather than part of the mathematical LMSV model.

---

# 4. How the two files work together

```text
                 lmsv_experiments.py
                         │
                         │ selects parameter configurations
                         ▼
                 LMSVParameters
                         │
                         ▼
               lmsv_monte_carlo.py
                         │
                         ├─ build fGn covariance
                         ├─ Cholesky factorization
                         ├─ simulate fGn
                         ├─ propagate fOU factor
                         ├─ compute σ(Y)
                         ├─ accumulate V_{0,T}
                         ├─ compute U_{0,T}
                         ├─ price LMSV call
                         ├─ estimate LMSV Delta
                         ├─ invert BS price
                         └─ compute hedging bias
                         │
                         ▼
                 numerical results
                         │
                         ▼
              CSV files and PNG figures
```

The first file answers:

> **How is one LMSV numerical specification evaluated?**

The second answers:

> **Which specifications should be evaluated in order to produce the numerical analysis of the thesis?**

---

# 5. Main numerical choices to remember

The most important implementation choices are:

- **Fractional Gaussian noise rather than reconstructed fBM paths:** the fOU recursion needs increments directly.
- **Cholesky simulation:** a transparent way of imposing the required finite-dimensional Gaussian covariance structure.
- **Small diagonal regularization:** prevents numerical Cholesky failures.
- **Bounded logistic volatility:** ensures positivity and compatibility with the theoretical assumptions.
- **`tanh` implementation of the logistic function:** reduces overflow risk and improves numerical stability.
- **Left-point discretization:** used in the fOU recursion and in the integrated-variance Riemann sum.
- **Batch processing:** reduces memory requirements without changing the estimator.
- **Conditional pricing rather than direct stock-path simulation:** exploits independence of the price and volatility noises and reduces the European-call problem to the distribution of U_{0,T}.
- **Brent's method:** robust numerical inversion for implied volatility.
- **Common random sample across strikes:** improves comparability of the hedging-bias results.
- **Fixed random seed:** makes the reported results reproducible.

---

# 6. Reproducing the numerical analysis

The external Python dependencies are

```text
numpy
scipy
matplotlib
```

The main numerical implementation can be run with

```bash
python src/lmsv_monte_carlo.py
```

and the complete collection of experiments with

```bash
python src/lmsv_experiments.py
```

The second command reproduces the numerical experiments and generates the corresponding result files.

---

# 7. Summary

The code simulates the **fractional volatility dynamics**, reduces each simulated volatility path to its **integrated volatility**, evaluates the corresponding **conditional Black--Scholes-type price and Delta**, averages these quantities by Monte Carlo, and finally compares the LMSV hedge with the **Black--Scholes implied-volatility hedge** through the hedging bias.
