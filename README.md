# Long-Memory Stochastic Volatility Monte Carlo Simulation

Python implementation accompanying the Master's thesis

> **Long-Memory Stochastic Volatility Models and Fractional Delta Hedging**

**Marco Guidi**  
MSc in Quantitative Finance  
Department of Statistical Sciences  
Alma Mater Studiorum – University of Bologna

---

## Overview

This repository contains the Python implementation developed for the numerical experiments presented in the Master's thesis *Long-Memory Stochastic Volatility Models and Fractional Delta Hedging*.

The code implements Monte Carlo simulation of a Long-Memory Stochastic Volatility (LMSV) model in which volatility is driven by a fractional Ornstein–Uhlenbeck process. The numerical analysis considers both European call options and arithmetic Asian call options.

For European calls, the implementation exploits the conditional Black–Scholes representation of the option price and Delta in terms of future integrated volatility. The numerical experiments investigate Monte Carlo convergence, time-grid stability, hedging bias across strikes, and sensitivity to the Hurst parameter.

The framework is subsequently extended to continuously monitored arithmetic Asian calls. Since their payoff depends on the asset-price path, the asset dynamics are simulated explicitly and the continuous arithmetic average is approximated on the numerical time grid using the trapezoidal rule. The Asian call Delta is estimated using both a pathwise estimator and centered finite differences with common random numbers. The numerical experiments include finite-difference bump-size sensitivity, time-grid stability, and a comparison of the effects of the Hurst parameter on European and Asian options.

The theoretical framework is based primarily on the following references:

- Comte, F., & Renault, É. (1998). *Long Memory Continuous-Time Models*.
- Zhao, Z., & Chronopoulou, A. (2023). *Fractional Delta Hedging*.

---

## Repository structure

```text
.
├── src/
│   ├── lmsv_monte_carlo.py
│   └── lmsv_experiments.py
├── results/
├── Monte_Carlo_Implementation_Guide.md
├── requirements.txt
└── README.md
```

The two Python scripts have distinct roles:

- `src/lmsv_monte_carlo.py` contains the implementation of the LMSV model and the Monte Carlo simulation functions.
- `src/lmsv_experiments.py` contains the numerical experiments used to generate and analyze the results reported in the thesis.

A detailed description of the numerical implementation and of the individual experiments is provided in `Monte_Carlo_Implementation_Guide.md`.

---

## Main features

The implementation includes:

- simulation of fractional Gaussian noise;
- simulation of the fractional Ornstein–Uhlenbeck volatility factor;
- bounded logistic volatility specification;
- simulation of future integrated volatility;
- Monte Carlo pricing of European call options through conditional Black–Scholes valuation;
- computation of the fractional Delta and comparison with the Black–Scholes Delta;
- validation in the Black–Scholes limit;
- Monte Carlo convergence analysis;
- time-grid stability analysis;
- analysis of hedging bias across strikes;
- sensitivity analysis with respect to the Hurst parameter;
- simulation of the asset dynamics and numerical approximation of the continuous arithmetic average;
- Monte Carlo pricing of continuously monitored arithmetic Asian call options;
- pathwise estimation of the Asian call Delta;
- finite-difference Delta estimation using common random numbers;
- finite-difference bump-size sensitivity analysis;
- time-grid stability analysis for the Asian option;
- comparison of European and Asian option results across different values of the Hurst parameter.

---

## Requirements

The code is written in Python 3.

The required packages are listed in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

---

## Running the code

The core LMSV implementation can be run with:

```bash
python src/lmsv_monte_carlo.py
```

The numerical experiments can be run with:

```bash
python src/lmsv_experiments.py
```

The experiment script generates the numerical outputs used in the analysis presented in the thesis.

---

## Reproducibility

This repository contains the source code used to reproduce the numerical experiments presented in the thesis.

Fixed random seeds are used in the numerical experiments where appropriate in order to make the reported results reproducible. Minor numerical differences may occur across software environments or numerical-library versions.

---

## Thesis

This repository accompanies the Master's thesis:

**Marco Guidi, *Long-Memory Stochastic Volatility Models and Fractional Delta Hedging***  
MSc in Quantitative Finance  
Alma Mater Studiorum – University of Bologna
