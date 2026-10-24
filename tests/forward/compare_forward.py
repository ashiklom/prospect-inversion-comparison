"""Compare every forward-model implementation against the NumPy reference.

Writes results/summary/forward_check.csv and exits non-zero on failure.
Tolerances: 1e-9 absolute for the same E1 mode (library E1 implementations
differ at ~1e-15 relative), 1e-8 for the independent `prosail` package.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/forward/out"
TOL = {"same": 1e-9, "prosail": 1e-8}

rows = []
for e1 in ("exact", "approx"):
    ref = pd.read_csv(OUT / f"reference_{e1}.csv")
    for f in sorted(OUT.glob(f"*_{e1}.csv")):
        impl = f.stem.rsplit("_", 1)[0]
        if impl == "reference":
            continue
        d = pd.read_csv(f).merge(ref, on=["case", "wavelength"], suffixes=("", "_ref"))
        for case, g in d.groupby("case", sort=False):
            dR = (g["R"] - g["R_ref"]).abs().max()
            dT = (g["T"] - g["T_ref"]).abs().max()
            tol = TOL["prosail"] if impl == "prosail" else TOL["same"]
            rows.append({"implementation": impl, "e1": e1, "case": case,
                         "max_abs_dR": dR, "max_abs_dT": dT, "tolerance": tol,
                         "pass": bool(max(dR, dT) <= tol)})

# Approximate vs exact E1, from the reference implementation
ex = pd.read_csv(OUT / "reference_exact.csv")
ap = pd.read_csv(OUT / "reference_approx.csv")
d = ap.merge(ex, on=["case", "wavelength"], suffixes=("_approx", "_exact"))
for case, g in d.groupby("case", sort=False):
    rows.append({"implementation": "reference (approx vs exact E1)", "e1": "approx-exact",
                 "case": case,
                 "max_abs_dR": (g["R_approx"] - g["R_exact"]).abs().max(),
                 "max_abs_dT": (g["T_approx"] - g["T_exact"]).abs().max(),
                 "tolerance": float("nan"), "pass": True})

res = pd.DataFrame(rows)
(ROOT / "results/summary").mkdir(parents=True, exist_ok=True)
res.to_csv(ROOT / "results/summary/forward_check.csv", index=False)
d[["case", "wavelength", "R_approx", "R_exact", "T_approx", "T_exact"]].to_csv(
    ROOT / "results/summary/e1_approx_vs_exact.csv.gz", index=False, float_format="%.8g")
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
if not res["pass"].all():
    print("FORWARD CHECK FAILED", file=sys.stderr)
    sys.exit(1)
print("All implementations agree with the reference.")
