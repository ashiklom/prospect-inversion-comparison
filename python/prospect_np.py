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


E1 = {"exact": e1_exact, "approx": e1_approx}


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
