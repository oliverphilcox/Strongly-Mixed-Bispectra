[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![ArXiv](https://img.shields.io/badge/arXiv-2302...-yellowgreen.svg)](https://google.com)

# Strongly-mixed bispectra in the π-σ model

Code accompanying *"Full-shape constraints of strongly-mixed bispectra with Planck data"* (Philcox, Pinol, Roest & Werth, in prep.) [[arXiv:XXXX.XXXXX]](https://arxiv.org/abs/XXXX.XXXXX).

This repository computes the tree-level bispectrum of the curvature perturbation for every cubic interaction of the Goldstone mode $\pi$ and an isocurvature scalar $\sigma$ at leading order in gradients, with the quadratic mixing $\rho\dot\pi\sigma$ treated **exactly**. It applies for any sound speeds $c_\pi,c_\sigma\leq1$, any mixing, and any mass with $m_{\rm eff}^2=m^2+\rho^2>0$, including light fields ($m_{\rm eff}<3H/2$) and tachyonic bare masses ($m^2<0$).

---

## What's here

| File | What it does |
|---|---|
| `strongmixing/modes.py` | `Modes`: the exact linear mode functions at one point $(c_s,\rho/H,m_{\rm eff}/H)$, the late-time amplitudes $\pi_\alpha(0)$ and the power-spectrum enhancement $\mathcal{R}$. |
| `strongmixing/bispectrum.py` | `Bispectrum`: the contact bispectrum shapes of the eight cubic vertices, for any triangle. |
| `Solver.py`, `Theory.py`, `Parameters.py` | `CosmoFlow`: an independent solver of the same in-in problem as a system of ODEs in time, used for validation. Its couplings follow its own conventions, set in `Theory.py`. |

### Setup

After rescaling space by $c_\sigma$, the quadratic Lagrangian is
$$a^{-3}\tilde{\mathcal{L}}_2=\tfrac12\left[\dot\pi_c^2-c_s^2(\tilde\partial_i\pi_c)^2/a^2\right]+\tfrac12\left[\dot\sigma_c^2-(\tilde\partial_i\sigma_c)^2/a^2-m^2\sigma_c^2\right]+\rho\dot\pi_c\sigma_c,$$
with $c_s=c_\pi/c_\sigma$, canonical fields $\pi_c=(f_\pi^2/c_s^{3/2})\pi$ and $\sigma_c=c_\sigma^{1/2}\sigma$, $f_\pi^4=2M_{\rm pl}^2|\dot H|c_\pi$, and $\zeta=-H\pi$.

### The interactions

Each vertex is a term $-\lambda(\partial_t^{n_1}X_1)(\partial_t^{n_2}X_2)(\partial_t^{n_3}X_3)$ or $-\lambda' a^{-2}(\partial_t^{n_1}Y_1)(\tilde\partial_iY_2)(\tilde\partial^iY_3)$ in $a^{-3}\tilde{\mathcal{L}}_3$:

| Vertex | Key | Shape returned per unit |
|---|---|---|
| $\dot\pi_c^3$ | `pidot3` | $\lambda H^2/\Delta_\zeta$ |
| $\sigma_c\dot\pi_c^2$ | `sigma_pidot2` | $\lambda H/\Delta_\zeta$ |
| $\dot\pi_c\sigma_c^2$ | `pidot_sigma2` | $\lambda/\Delta_\zeta$ |
| $\dot\pi_c\dot\sigma_c^2$ | `pidot_sigmadot2` | $\lambda H^2/\Delta_\zeta$ |
| $\sigma_c^3$ | `sigma3` | $\lambda H^{-1}/\Delta_\zeta$ |
| $\dot\pi_c(\tilde\partial_i\pi_c)^2/a^2$ | `pidot_dpi2` | $\lambda' H^2/\Delta_\zeta$ |
| $\sigma_c(\tilde\partial_i\pi_c)^2/a^2$ | `sigma_dpi2` | $\lambda' H/\Delta_\zeta$ |
| $\dot\sigma_c\tilde\partial_i\pi_c\tilde\partial^i\sigma_c/a^2$ | `sigmadot_dpi_dsigma` | $\lambda' H^2/\Delta_\zeta$ |

The shape is $S=(k_1k_2k_3)^2B_\zeta/[(2\pi)^4\Delta_\zeta^4]$, where $\Delta_\zeta^2=\mathcal{R}H^4/(4\pi^2f_\pi^4)$ is the full (mixed) power spectrum. With the convention $\tfrac{5}{18}B_\zeta/(P_1P_2P_3)^{2/3}=f_{\rm NL}S^{\rm th}$ of the [Planck binned shape](https://github.com/oliverphilcox/Binned-Bispectrum-Shape), $\tfrac{5}{18}B_\zeta/(P_1P_2P_3)^{2/3}=\tfrac{10}{9}S$.

---

## Method (brief)

The mode functions are written as superpositions of plane waves, $F_\alpha(x)=\int\mathrm{d}s\,e^{isx}f_\alpha(s)$ with $\pi_\alpha=(1-x\partial_x)F_\alpha$ and $\sigma_\alpha=-(x^2H/\rho)(\partial_x^2+c_s^2)F_\alpha$, whose weight satisfies a Heun equation ($c_s\neq1$) or an associated Legendre equation ($c_s=1$). The frequency integrals run along straight rays from the two light cones, $s=c_s+it$ and $s=1+it$, which coincide at $c_s=1$. For $c_s\neq1$ the weights are obtained from their Taylor series at each cone and transported along the rays with the Heun equation; for $c_s=1$ they are hypergeometric functions. Every bispectrum is then a contact diagram whose legs are the bulk-to-boundary propagators $G^X_k(\tau)=(H^2/2k^3)\sum_\alpha\pi_\alpha(0)X_\alpha^*(-k\tau)$, so all mixing insertions are resummed. The time integral runs along real time from $\tau_*$ (shortly before horizon crossing) to the end of inflation, where the integrand contains at least one commutator and converges for every $m_{\rm eff}>0$, and along $\tau_*+i\eta$ at earlier times, which implements the $i\epsilon$ prescription. Along real time, the imaginary parts of the propagators are obtained directly from the fourth-order equation of motion.

The shapes are computed in the late-time limit. For very light fields this is reached slowly: the error from a finite number $N_{\rm end}$ of $e$-folds after horizon exit scales as $e^{-(3/2-\nu_{\rm eff})N_{\rm end}}$, about $1\%$ at $m_{\rm eff}=H/2$ for $N_{\rm end}\approx55$, so we recommend $m_{\rm eff}\gtrsim H/2$.

---

## Installation

```bash
pip install numpy scipy mpmath
```

`CosmoFlow` also needs `matplotlib`.

No installation is needed: `strongmixing` is imported from a clone of this repository.

---

## Quickstart

```python
from strongmixing import Modes, Bispectrum

# Mode functions at c_s = 0.5, rho/H = 3.16, m_eff/H = 3.5
modes = Modes(c_s=0.5, rho=3.16, m_eff=3.5)
print(modes.R)

# All eight shapes at one triangle, or a subset
bispectrum = Bispectrum(modes)
S = bispectrum.shapes(1.0, 0.8, 0.6)
S_pidot3 = bispectrum.shapes(1.0, 0.8, 0.6, names=['pidot3'])['pidot3']
```

A `Bispectrum` is reused for any number of triangles at the same parameter point.
