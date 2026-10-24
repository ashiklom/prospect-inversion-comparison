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

# PyTensor compiled function
N_ = pt.scalar("N")
C_ = pt.vector("C")
fns = {}
for e1 in ("exact", "approx"):
    R_, T_ = prospect_pt.prospect(N_, C_, model.K, model.talf, model.t12, model.t21,
                                  e1=e1, want_t=True)
    fns[e1] = pytensor.function([N_, C_], [R_, T_])

def frame(rows):
    return pd.concat(rows, ignore_index=True)

for e1 in ("exact", "approx"):
    ref, ptr = [], []
    for c in cases["cases"]:
        R, T = model(c["N"], c["C"], e1=e1)
        ref.append(pd.DataFrame({"case": c["name"], "wavelength": model.wavelength, "R": R, "T": T}))
        R, T = fns[e1](c["N"], np.asarray(c["C"], dtype=float))
        ptr.append(pd.DataFrame({"case": c["name"], "wavelength": model.wavelength, "R": R, "T": T}))
    frame(ref).to_csv(outdir / f"reference_{e1}.csv", index=False, float_format="%.17g")
    frame(ptr).to_csv(outdir / f"pytensor_{e1}.csv", index=False, float_format="%.17g")

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
