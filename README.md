# 🧠 Reservoir Neural Networks with Fourier Layer
### Advanced Reservoir Computing Techniques for Chaotic Time Series Prediction
**By [ACS Lab, ITMO University](https://iai.itmo.ru/)** · 2025

Official repository for the paper:
> **A. Kovantsev, R. Vysotskiy (2025). _Advanced Reservoir Neural Network Techniques for Chaotic Time Series Prediction._ SSRN 5481760.**

---

## 📘 Overview

We present **ESN‑F** — the **Echo State Network (ESN)** enhanced with **Fourier features** and **polynomial expansion** for forecasting **chaotic / nonlinear** time series. The approach keeps the **reservoir untrained** and learns only a **ridge readout**, while enriching inputs with **periodic (sin/cos)** and **nonlinear** bases that improve long‑horizon stability.

Use cases include **finance/economics**, **risk modeling**, and other **non‑stationary** domains.

---

## Contents

* [Why this project?](#why-this-project)
* [Install](#install)
* [Quickstart](#quickstart)
* [Generative (multi-step) forecasting](#generative-multi-step-forecasting)
* [Multivariate example](#multivariate-example)
* [Background & Theory](#background--theory)

  * [Echo State Network (leaky ESN)](#echo-state-network-leaky-esn)
  * [Fourier & Polynomial feature blocks (FAN)](#fourier--polynomial-feature-blocks-fan)
  * [Ridge readout & objective](#ridge-readout--objective)
  * [Forecasting strategies](#forecasting-strategies)
  * [Why it helps with chaos & non‑stationarity](#why-it-helps-with-chaos--non-stationarity)
* [Predictability metrics (optional)](#predictability-metrics-optional)
* [📐 Predictability features (formulas)](#-predictability-features-formulas)
* [Hyperparameters](#hyperparameters)
* [Complexity & scaling](#complexity--scaling)
* [Reproducibility](#reproducibility)
* [Figures (placeholders)](#figures-placeholders)
* [📊 Experimental Results (from the paper)](#-experimental-results-from-the-paper)
* [Project layout](#project-layout)
* [Contributing](#contributing)
* [License & citation](#license--citation)
* [Troubleshooting / FAQ](#troubleshooting--faq)

---

## Why this project?

Long‑horizon forecasting on chaotic / weakly stationary signals is tricky: standard RNNs tend to drift, while purely statistical baselines miss nonlinear structure. **EnhancedESN_FAN** keeps a **random, untrained reservoir** for rich dynamics and augments the readout with **deterministic Fourier harmonics** and **polynomial features**. The linear readout is trained via **ridge regression**, keeping training fast, convex, and robust.

---

## Install

```bash
# From source (recommended for now)
git clone https://github.com/CapitalistGeorge/chaotic_library.git
cd chaotic_library

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

python -m pip install -U pip

# Option 1: install dependencies only (for running notebooks / scripts)
pip install -r requirements.txt

# Option 2: install the package in editable mode (preferred while developing)
pip install -e .
```

*Planned:* after the API is frozen we’ll publish to PyPI, so you can `pip install esn-fan` (package name TBD). The import path in the examples below assumes a module `enhanced_esn_fan.py` at the project root; adjust if packaged differently.

---

## Quickstart

```python
import numpy as np
from chaotic_library import EnhancedESN_FAN  # adjust import path if packaged differently

# 1) Build training data (shape: [n_timesteps, input_dim])
# Univariate example: input_dim=1 → X is 2D with one column, y is 1D
T = 1200
noise = 0.1 * np.random.randn(T)
signal = np.sin(2*np.pi*np.arange(T)/50) + 0.25*np.sin(2*np.pi*np.arange(T)/7) + noise

X = signal[:-1].reshape(-1, 1)   # features are previous value(s)
y = signal[1:]                   # next-step target

# 2) Initialize Enhanced ESN + FAN features (Fourier + polynomial)
esn = EnhancedESN_FAN(
    input_dim=1,          # number of input features per timestep (columns of X)
    reservoir_size=800,
    spectral_radius=0.95,
    sparsity=0.1,
    ridge_alpha=1e-2,
    leaking_rate=0.3,
    poly_order=2,
    fan_terms=8,
    random_state=42,
    clip_value=3.0,
)

# 3) Fit and one-step-ahead predictions (teacher forcing / open loop)
esn.fit(X, y)
y_hat = esn.predict(X[:100])        # shape: (100,) for univariate target
print("y_hat shape:", np.asarray(y_hat).shape)
```

---

## Generative (multi-step) forecasting

```python
# Seed with the last observed input row (shape: [1, input_dim])
seed = X[-1:].copy()

# Produce next 300 steps autoregressively
future = esn.predict(seed, generative_steps=300)   # shape: (300,) for univariate

# Concatenate history + forecast for plotting
full = np.concatenate([signal, future.ravel()])
```

---

## Multivariate example

```python
import numpy as np
from chaotic_library import EnhancedESN_FAN

# Suppose you have 3 exogenous drivers + the main signal → input_dim=4
n = 2000
main = np.sin(2*np.pi*np.arange(n)/30) + 0.05*np.random.randn(n)
x1 = np.cos(2*np.pi*np.arange(n)/100)
x2 = np.sin(2*np.pi*np.arange(n)/7)
x3 = 0.01*np.arange(n)  # slow trend proxy

# Features: use current drivers + lagged main as input; predict next main
X = np.column_stack([main[:-1], x1[:-1], x2[:-1], x3[:-1]])  # shape (n-1, 4)
y = main[1:]                                               # shape (n-1,)

model = EnhancedESN_FAN(
    input_dim=4,
    reservoir_size=500,
    spectral_radius=0.9,
    sparsity=0.1,
    ridge_alpha=0.1,
    leaking_rate=0.4,
    poly_order=2,
    fan_terms=6,
    random_state=7,
)

model.fit(X, y)
pred = model.predict(X[:128])              # one-step predictions (teacher forcing)
gen  = model.predict(X[-1:], generative_steps=200)  # recursive forecast
```

---

## Background & Theory

### Echo State Network (leaky ESN)

Reservoir state \(\mathbf{x}_t \in \mathbb{R}^{N_r}\) evolves under a fixed random dynamical system:

$$
\begin{aligned}
\tilde{\mathbf{x}}_t &= \tanh\left( \mathbf{W} \mathbf{x}_{t-1} + \mathbf{W}_{\text{in}} [1; u_t] \right), \\
\mathbf{x}_t &= (1-\alpha) \mathbf{x}_{t-1} + \alpha \tilde{\mathbf{x}}_t,
\end{aligned}
$$

where:
- \(u_t\) is the input (e.g., components of \(X_t\))
- \([1; u_t]\) denotes a bias‑augmented input
- \(\alpha\) is the **leaking rate** (`leaking_rate` parameter)

To satisfy the **echo state property** (state forgets initial conditions), scale the reservoir so that its spectral radius \(\rho(\mathbf{W})\) is near 1 (practically 0.7–1.2 with leakage; `spectral_radius` parameter).

We collect features at time \(t\) by concatenating the reservoir state with deterministic blocks:

$$
\mathbf{z}_t = \big[ \mathbf{x}_t \mid \phi_{\text{poly}}(u_t) \mid \phi_{\text{Fourier}}(u_t) \big] \in \mathbb{R}^{D}
$$

**Notation:**
- \(\mathbf{W}\): reservoir weight matrix
- \(\mathbf{W}_{\text{in}}\): input weight matrix  
- \(\phi_{\text{poly}}\): polynomial features
- \(\phi_{\text{Fourier}}\): Fourier features
- \(D\): total feature dimension

### Fourier & Polynomial feature blocks (FAN)

* **PolynomialFeatures** of degree (d) (=`poly_order`): ([u_t, u_t^2, \dots, u_t^d]) per input dimension (no extra bias term; bias provided separately).
* **Fourier (FAN) features** with harmonics (k=1..K) (=`fan_terms`): for each input dimension, compute (\sin(2\pi k X)) and (\cos(2\pi k X)). These inject periodic structure explicitly, so the reservoir does not have to "discover" it from scratch.

### Ridge readout & objective

Only the final linear readout (\mathbf{W}*{\text{out}} \in \mathbb{R}^{D\times m}) is trained via ridge regression:
[
\min*{\mathbf{W}*{\text{out}}}; \big|\mathbf{Y} - \mathbf{Z}\mathbf{W}*{\text{out}}\big|*2^2
;+; \lambda \big|\mathbf{W}*{\text{out}}\big|*2^2,
\quad\Rightarrow\quad
\mathbf{W}*{\text{out}} = (\mathbf{Z}^\top\mathbf{Z} + \lambda \mathbf{I})^{-1}\mathbf{Z}^\top\mathbf{Y}.
]
Columns of (\mathbf{Z}) should be standardized for numerical stability (the implementation uses `StandardScaler`).

### Forecasting strategies

* **Teacher forcing / open loop (default in `predict(X)`):** one‑step predictions using the provided inputs.
* **Generative / recursive (`predict(seed, generative_steps=m)`):** feed back model outputs as inputs to generate future steps.
* **Hybrid (future option):** recursive core with direct corrections for selected horizons.

### Why it helps with chaos & non‑stationarity

* Reservoir provides a **rich, fading memory** of nonlinear histories.
* Fourier layer **anchors** periodic structure → less burden on the reservoir.
* Polynomial bias **stabilizes** local trends and offsets.
* Ridge readout **tames variance** and keeps training convex & fast.

---

## Predictability metrics (optional)

You can compute these to **cluster series by predictability** and adapt hyperparameters:

* **Hurst exponent (H)** — persistence (>0.5) vs. anti‑persistence (<0.5)
* **Correlation dimension (D₂)** — attractor dimension (Grassberger–Procaccia)
* **Max Lyapunov exponent (λₘₐₓ)** — sensitivity to initial conditions
* **Kolmogorov–Sinai entropy (KSE)** — information production rate
* **# Prevailing harmonics** — count strong spectral peaks (e.g., via periodogram)

Use the cluster to pick `reservoir_size`, `spectral_radius`, and `fan_terms`. For highly chaotic signals (large λₘₐₓ), prefer slightly lower `spectral_radius` and stronger regularization (`ridge_alpha`).

---

## 📐 Predictability features (formulas)

NOTATION:
  x̄_τ    - sample mean on window of length τ
  θ(·)   - Heaviside step function
  ρ(i,j) - distance in reconstructed phase space (delay embedding optional)
  x'_i   = x_i - x_{i-1}  (first difference)

HURST EXPONENT
H = ln( R(τ) / S(τ) ) / ln(α·τ)                                 (6)

where:
R(τ) = max[1≤t≤τ] [ Σ[i=1 to t] (x_i - x̄_τ) ] 
       - min[1≤t≤τ] [ Σ[i=1 to t] (x_i - x̄_τ) ]                 (7)
       
S(τ) = √[ (1/τ) · Σ[t=1 to τ] (x_t - x̄_τ)² ]                   (8)

Note: in classical R/S analysis, H is the slope of ln(R/S) vs ln τ 
(i.e., α=1). Including a constant α is equivalent up to offset.

KOLMOGOROV-SINAI ENTROPY (KSE)
Definition via entropy-rate upper bound:

h_μ(T,ξ) = - lim[n→∞] (1/n) 
           × Σ[i₁,...,iₙ] μ( T⁻¹C_i₁ ∩ ... ∩ T⁻ⁿC_iₙ ) · ln μ(...)  (9)

h_μ^KS(T) = sup[ξ] h_μ(T,ξ)                                       (10)

CORRELATION DIMENSION
d₂ = lim[r→0] lim[m→∞] ln C(r) / ln r                           (11)

where:
C(r) = 1 / [m(m-1)] 
       × Σ[i=1 to m] Σ[j=i+1 to m] θ( r - ρ(i,j) )              (12,13)

---

## Hyperparameters

| Parameter         | Meaning                                             | Typical range / tips                |
| ----------------- | --------------------------------------------------- | ----------------------------------- |
| `reservoir_size`  | Number of reservoir units                           | 300–2000                            |
| `spectral_radius` | Spectral radius after scaling                       | 0.7–1.2 with leakage                |
| `sparsity`        | Fraction of **zeroed** connections (mask threshold) | 0.7–0.95 for very sparse reservoirs |
| `leaking_rate`    | Leaky integrator rate                               | 0.1–0.5 for longer memory           |
| `ridge_alpha`     | Ridge regularization strength                       | 1e−6–1e0                            |
| `poly_order`      | Polynomial degree (no bias term)                    | 1–3                                 |
| `fan_terms`       | #Fourier harmonics per input dimension              | 3–12                                |
| `clip_value`      | Clip for scaled inputs in generative mode           | 2–5                                 |
| `random_state`    | Seed                                                | set for reproducibility             |

---

## Complexity & scaling

* **State update:** (O(T,N_r,s)) with sparsity fraction (s) (dense → (O(T,N_r^2))).
* **Readout training:** build (\mathbf{Z}\in\mathbb{R}^{T\times D}); solve ridge via Cholesky/QR: ~(O(D^3)) (usually (D \ll T)).
* **Memory:** (O(TD)) if keeping all features; use chunked/online solvers for very long series.

---

## Reproducibility

* Fix `random_state` for weights and reservoirs.
* Standardize inputs and feature matrix consistently across train/forecast.
* Log: hyperparameters, seeds, and package versions.
* Provide notebooks that mirror experiments and regenerate figures.

---

## Figures (placeholders)

Put images in `docs/figures/` (SVG/PNG). Filenames below are suggestions; feel free to rename.

**Model architecture** <img src="docs/figures/fig-architecture-esnf.svg" width="760" alt="ESN-FAN architecture: input → reservoir (leaky ESN) → concat Fourier & poly → ridge readout"/>

**Reservoir dynamics (phase portrait)** <img src="docs/figures/fig-reservoir-dynamics.png" width="760" alt="Reservoir state trajectories and fading memory with different leak rates"/>

**Ablation: effect of feature blocks** <img src="docs/figures/fig-ablation-fourier-poly.png" width="760" alt="MAPE/RMSE across ESN, ESN+Poly, ESN+Fourier, ESN+Fourier+Poly"/>

**Predictability clustering** <img src="docs/figures/fig-predictability-clusters.svg" width="760" alt="Series clustered by H, D2, λmax, KSE, #harmonics; hyperparameter recipes per cluster"/>

**Long-horizon forecast vs. truth** <img src="docs/figures/fig-forecast-long-horizon.png" width="760" alt="Ground truth vs ESN-FAN forecast with prediction intervals; error growth comparison"/>

*(If you prefer pure Markdown images, you can also use: `![Architecture](docs/figures/fig-architecture-esnf.svg)` and similar.)*

---

## 📊 Experimental Results (from the paper)

### M4 (clustered by predictability, MAPE % ↓)

| Cluster |    ESN‑F |      ESN | LGBM | Prophet |   SSA |
| ------- | -------: | -------: | ---: | ------: | ----: |
| Good    | **3.44** |     3.56 | 3.72 |    6.86 | 18.03 |
| Bad     |     5.26 | **5.19** | 5.39 |    8.57 | 20.05 |

In the **Bad** cluster, ESN‑F beats LGBM by ≥1 pp in **27%** of series (LGBM better in **15%**; remainder negligible).

### Moscow Real Estate (weekly, MAPE % ↓)

| Model     |     MAPE |
| --------- | -------: |
| **ESN‑F** | **2.56** |
| ESN       |     3.19 |
| LGBM      |     7.18 |

Chaotic traits of the real‑estate series (for interpretation): Hurst **0.65**, Noise **0.99**, Corr. dimension **1.33**, max Lyapunov **0.01**, **KSE** **1.84**, (N_{Fh}=30).

---

## Project layout

```text
.
├── src/
│   └── chaotic_library/
│       ├── __init__.py            # public API (EnhancedESN_FAN, chaos measures, version, etc.)
│       ├── enhanced_esn_fan.py    # ESN-FAN implementation
│       └── chaotic_measures.py    # Hurst, Lyapunov, entropy, dimensionality, etc.
├── tests/                         # unit tests
├── .github/workflows/             # CI (linting, tests)
├── requirements.txt               # runtime/dev dependencies
├── pyproject.toml                 # packaging metadata (build system, project info)
├── README.md
└── LICENSE
```

---

## Contributing

* Run linters/formatters before committing (e.g., `ruff check .` / `ruff format .`).
* Add/extend tests in `tests/`.
* For new feature blocks, include a minimal notebook demo.
* Keep figures reproducible from notebooks where possible.

---

## License & citation

**License:** MIT — see [`LICENSE`](./LICENSE).

**Citation (placeholder):** If you use this repository, please cite the corresponding preprint/paper.

```
@misc{esn_fan_2025,
  title   = {Enhanced Echo State Network with Fourier Analysis Network (FAN) Features},
  author  = {Kovantsev, A. and Vysotskiy, R.},
  year    = {2025},
  note    = {preprint},
  howpublished = {URL: add when available}
}
```

---

## Troubleshooting / FAQ

**Q: My recursive (generative) forecast saturates or explodes.**
A: Increase `ridge_alpha`, decrease `spectral_radius`, and consider a slightly larger `clip_value` (2–5). Also try lowering `leaking_rate` for longer memory.

**Q: Shapes?**
A: `X` must be 2D: `(n_timesteps, input_dim)`. For univariate, reshape with `reshape(-1, 1)`. `y` can be 1D (univariate) or 2D (multi‑output).

**Q: Scaling consistency between train and predict?**
A: The model uses internal scalers. Ensure that polynomial and Fourier features at prediction time are computed in a way consistent with training. If you modify the code, apply the same input scaling before feature generation in all paths (teacher forcing and generative).
