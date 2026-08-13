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

The code implements Monte Carlo simulations for a Long-Memory Stochastic Volatility (LMSV) model driven by a fractional Ornstein–Uhlenbeck volatility factor. It is used to investigate European option pricing and Delta hedging under long-memory stochastic volatility.

The theoretical framework is based primarily on the following references:

- Comte, F., & Renault, É. (1998). *Long Memory Continuous-Time Models*.
- Zhao, Z., & Chronopoulou, A. (2023). *Fractional Delta Hedging*.

---

## Repository structure

```
.
├── src/               # Python source code
├── results/           # Simulation outputs
├── requirements.txt   # Python dependencies
├── LICENSE
└── README.md
```

---

## Main features

The implementation includes:

- simulation of the fractional Ornstein–Uhlenbeck volatility process;
- bounded logistic volatility specification;
- Monte Carlo simulation of the asset price dynamics;
- pricing of European call options;
- Delta hedging experiments under stochastic volatility.

---

## Requirements

The code is written in Python 3.

Required packages are listed in

```
requirements.txt
```

and can be installed with

```bash
pip install -r requirements.txt
```

---

## Reproducibility

This repository contains the source code used to reproduce the numerical experiments presented in the thesis.

After installing the required Python packages, the scripts can be executed independently to reproduce the reported numerical results.
