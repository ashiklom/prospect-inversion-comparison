"""Reference NumPy implementation of the PROSPECT family of leaf RTMs.

This is the canonical implementation against which every other backend
(Stan, Nimble, base R, Julia, PyTensor) is checked in `tests/forward`.

The model is version-agnostic: a PROSPECT version is just a table of
specific absorption coefficients (one column per constituent) plus a
refractive index. See `config/models.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import exp1

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Exponential integral E1
# ---------------------------------------------------------------------------


def e1_exact(x):
    """Exact E1(x) (Cephes, via SciPy)."""
    return exp1(x)


def e1_approx(x):
    """Swamee & Ohija (2003) closed-form approximation to E1(x), x > 0.

    Cheap and differentiable in any DSL. Relative error is ~1e-3 at worst,
    which can make the transmissivity slightly negative at very large k
    (clamped to 0 in `transmissivity`).
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):  # b -> inf (E1 -> 0) for x > ~90
        a = np.log((0.56146 / x + 0.65) * (1 + x))
        b = x**4 * np.exp(7.7 * x) * (2 + x) ** 3.7
        return (a**-7.7 + b) ** -0.13


# NAG S13AAF Chebyshev expansions (Horner form), as used in the original
# PROSPECT Fortran (PEcAnRTM/src/RTM/prospect/prospect.expint.f90).
# Max relative error vs. Cephes ~2e-12; fixed-length polynomials, so the
# evaluation is branch-light and vectorizes well.
E1_FAST_LO = (  # 0 < x <= 4, in t = x/2 - 1;  E1 = poly(t) - log(x)
    -3.60311230482612224e-13, 3.46348526554087424e-12, -2.99627399604128973e-11,
    2.57747807106988589e-10, -2.09330568435488303e-9, 1.59501329936987818e-8,
    -1.13717900285428895e-7, 7.55292885309152956e-7, -4.64980751480619431e-6,
    2.63830365675408129e-5, -1.37089870978830576e-4, 6.47686503728103400e-4,
    -2.76060141343627983e-3, 1.05306034687449505e-2, -3.57191348753631956e-2,
    1.07774527938978692e-1, -2.96997075145080963e-1, 8.64664716763387311e-1,
    7.42047691268006429e-1)
E1_FAST_HI = (  # x > 4, in t = 14.5/(x + 3.25) - 1;  E1 = exp(-x) * poly(t) / x
    -1.62806570868460749e-12, -8.95400579318284288e-13, -4.08352702838151578e-12,
    -1.45132988248537498e-11, -8.35086918940757852e-11, -2.13638678953766289e-10,
    -1.10302431467069770e-9, -3.67128915633455484e-9, -1.66980544304104726e-8,
    -6.11774386401295125e-8, -2.70306163610271497e-7, -1.05565006992891261e-6,
    -4.72090467203711484e-6, -1.95076375089955937e-5, -9.16450482931221453e-5,
    -4.05892130452128677e-4, -2.14213055000334718e-3, -1.06374875116569657e-2,
    -8.50699154984571871e-2, 9.23755307807784058e-1)


def _horner(coefs, t):
    y = np.zeros_like(t)
    for c in coefs:
        y = y * t + c
    return y


def e1_fast(x):
    """Fast, accurate E1(x), x > 0 (NAG S13AAF Chebyshev expansions)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lo = _horner(E1_FAST_LO, 0.5 * x - 1.0) - np.log(x)
        hi = np.exp(-x) * _horner(E1_FAST_HI, 14.5 / (x + 3.25) - 1.0) / x
    return np.where(x <= 4.0, lo, hi)


E1 = {"exact": e1_exact, "fast": e1_fast, "approx": e1_approx}


# ---------------------------------------------------------------------------
# Interface transmissivity (Stern 1964; Allen 1973)
# ---------------------------------------------------------------------------


def tav(theta_deg, n):
    """Average transmissivity of a dielectric plane surface.

    `theta_deg` is the maximum incidence angle (degrees), `n` the refractive
    index (array). Port of `tav_abs` in rrtm/R/gpm.R.
    """
    n = np.asarray(n, dtype=float)
    theta = np.deg2rad(theta_deg)
    if theta_deg == 0:
        return 4 * n / (n + 1) ** 2
    r2 = n**2
    rp = r2 + 1
    rm = r2 - 1
    a = (n + 1) ** 2 / 2
    k = -((r2 - 1) ** 2) / 4
    sa = np.sin(theta)
    if theta_deg == 90:
        b1 = 0.0
    else:
        b1 = np.sqrt((sa**2 - rp / 2) ** 2 + k)
    b2 = sa**2 - rp / 2
    b = b1 - b2
    ts = (k**2 / (6 * b**3) + k / b - b / 2) - (k**2 / (6 * a**3) + k / a - a / 2)
    tp1 = -2 * r2 * (b - a) / rp**2
    tp2 = -2 * r2 * rp * np.log(b / a) / rm**2
    tp3 = r2 * (1 / b - 1 / a) / 2
    tp4 = (
        16 * r2**2 * (r2**2 + 1)
        * np.log((2 * rp * b - rm**2) / (2 * rp * a - rm**2))
        / (rp**3 * rm**2)
    )
    tp5 = 16 * r2**3 * (1 / (2 * rp * b - rm**2) - 1 / (2 * rp * a - rm**2)) / rp**3
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    return (ts + tp) / (2 * sa**2)


# ---------------------------------------------------------------------------
# Generalized plate model
# ---------------------------------------------------------------------------


def transmissivity(k, e1="exact"):
    """Transmissivity of a compact layer with absorption coefficient k."""
    k = np.asarray(k, dtype=float)
    kpos = np.where(k > 0, k, 1.0)
    tau = (1 - kpos) * np.exp(-kpos) + kpos**2 * E1[e1](kpos)
    tau = np.where(k > 0, tau, 1.0)
    return np.maximum(tau, 0.0)


def gpm(k, N, talf, t12, t21, e1="exact"):
    """Generalized plate model: reflectance and transmittance of N plates.

    Follows rrtm/R/gpm.R (Allen et al. 1969 single plate + Stokes 1862
    N-layer solution).
    """
    tau = transmissivity(k, e1)

    # Top layer
    ralf = 1 - talf
    r12 = 1 - t12
    r21 = 1 - t21
    denom = 1 - r21**2 * tau**2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    # Elementary layer
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    # Stokes N-layer solution
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.sqrt(np.maximum((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), 0))
        va = (1 + r**2 - t**2 + D) / (2 * r)
        vb = (1 - r**2 + t**2 + D) / (2 * t)
        vbNN = vb ** (N - 1)
        denomx = va**2 * vbNN**2 - 1
        Rsub = va * (vbNN**2 - 1) / denomx
        Tsub = vbNN * (va**2 - 1) / denomx
    # No absorption (r + t = 1): Kubelka-Munk-like limit
    noabs = (r + t) >= 1
    Tsub_na = t / (t + (1 - t) * (N - 1))
    Rsub = np.where(noabs, 1 - Tsub_na, Rsub)
    Tsub = np.where(noabs, Tsub_na, Tsub)
    # Total absorption
    opaque = (t <= 0) | (r <= 0)
    Rsub = np.where(opaque & ~noabs, 0.0, Rsub)
    Tsub = np.where(opaque & ~noabs, 0.0, Tsub)

    denomy = 1 - Rsub * r
    R = Ra + Ta * Rsub * t / denomy
    T = Ta * Tsub / denomy
    return R, T


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def load_models(root: Path = ROOT) -> dict:
    with open(root / "config" / "models.json") as f:
        return json.load(f)


class Prospect:
    """A PROSPECT version: coefficient table + constituents.

    Parameters
    ----------
    version : key in config/models.json (e.g. "prospectd")
    wavelengths : optional subset of wavelengths (nm) to evaluate at
    """

    def __init__(self, version: str = "prospectd", wavelengths=None, root: Path = ROOT):
        spec = load_models(root)[version]
        self.version = version
        self.constituents = list(spec["constituents"])
        coef = pd.read_csv(root / spec["coefficients"])
        if wavelengths is not None:
            coef = coef.set_index("wavelength").loc[np.asarray(wavelengths)].reset_index()
        self.wavelength = coef["wavelength"].to_numpy()
        self.K = coef[self.constituents].to_numpy()
        self.talf = coef["talf"].to_numpy()
        self.t12 = coef["t12"].to_numpy()
        self.t21 = coef["t21"].to_numpy()

    def __call__(self, N, C, e1="exact"):
        """Return (R, T) for structure parameter N and constituent vector C
        (ordered as `self.constituents`, or a dict keyed by name)."""
        if isinstance(C, dict):
            C = np.array([C[c] for c in self.constituents])
        k = self.K @ np.asarray(C, dtype=float) / N
        return gpm(k, N, self.talf, self.t12, self.t21, e1=e1)
