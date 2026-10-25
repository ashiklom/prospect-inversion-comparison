"""Evaluate the NumPy reference, PyTensor and prosail PROSPECT-D on the shared cases."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from prospect_np import Prospect  # noqa: E402
import prospect_pt  # noqa: E402

cases = json.load(open(ROOT / "tests/forward/cases.json"))
outdir = ROOT / "tests/forward/out"
outdir.mkdir(exist_ok=True)
model = Prospect("prospectd")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

# PyTensor compiled functions: default (numba) backend and JAX backend
N_ = pt.scalar("N")
C_ = pt.vector("C")
E1_MODES = ("exact", "fast", "approx")
fns = {}
for e1 in E1_MODES:
    R_, T_ = prospect_pt.prospect(N_, C_, model.K, model.talf, model.t12, model.t21,
                                  e1=e1, want_t=True)
    fns[("pytensor", e1)] = pytensor.function([N_, C_], [R_, T_])
    fns[("pytensor-jax", e1)] = pytensor.function([N_, C_], [R_, T_], mode="JAX")

def frame(rows):
    return pd.concat(rows, ignore_index=True)

for e1 in E1_MODES:
    out = {"reference": [], "pytensor": [], "pytensor-jax": []}
    for c in cases["cases"]:
        R, T = model(c["N"], c["C"], e1=e1)
        out["reference"].append(pd.DataFrame({"case": c["name"], "wavelength": model.wavelength, "R": R, "T": T}))
        for impl in ("pytensor", "pytensor-jax"):
            R, T = fns[(impl, e1)](c["N"], np.asarray(c["C"], dtype=float))
            out[impl].append(pd.DataFrame({"case": c["name"], "wavelength": model.wavelength,
                                           "R": np.asarray(R), "T": np.asarray(T)}))
    for impl, rows in out.items():
        frame(rows).to_csv(outdir / f"{impl}_{e1}.csv", index=False, float_format="%.17g")

# Independent implementation: prosail (exact E1 only)
import prosail  # noqa: E402
rows = []
for c in cases["cases"]:
    Cab, Car, Canth, Cbrown, Cw, Cm = c["C"]
    wv, R, T = prosail.run_prospect(c["N"], Cab, Car, Cbrown, Cw, Cm, ant=Canth,
                                    prospect_version="D")
    rows.append(pd.DataFrame({"case": c["name"], "wavelength": wv.astype(int), "R": R, "T": T}))
frame(rows).to_csv(outdir / "prosail_exact.csv", index=False, float_format="%.17g")
print("python forward tests written")
