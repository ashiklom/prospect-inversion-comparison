"""Build the coefficient tables and validation data used by every backend.

Inputs (sibling repositories; override with environment variables):
  RTM_JL_DIR     ../RTM.jl          -> data-raw/kmat_*.dat, refractive_*.dat
  SPECTRA_DB_DIR ../spectra_db      -> lopex/, angers/ csvy files

Outputs:
  data/coefficients/{prospectd,prospect5,prospect4}.csv
      wavelength, n, talf, t12, t21, <constituent columns>
  data/validation/traits.csv     one row per validation sample
  data/validation/spectra.csv    long format: sample_id, replicate, wavelength, R, T
  data/inputs/<version>/<sample_id>.csv
      fit-ready table: wavelength, R, T, talf, t12, t21, <K columns>
      (only wavelengths where the observed reflectance is finite)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from prospect_np import Prospect, tav  # noqa: E402

RTM_JL = Path(os.environ.get("RTM_JL_DIR", ROOT.parent / "RTM.jl"))
SPECTRA_DB = Path(os.environ.get("SPECTRA_DB_DIR", ROOT.parent / "spectra_db"))

COEF_SOURCES = {
    # output name: (kmat file, kmat columns, refractive index file)
    "prospectd": ("kmat_pd.dat", ["Cab", "Car", "Canth", "Cbrown", "Cw", "Cm"], "refractive_pd.dat"),
    "prospect5": ("kmat_p5.dat", ["Cab", "Car", "Cbrown", "Cw", "Cm"], "refractive_p45.dat"),
    "prospect4": ("kmat_p4.dat", ["Cab", "Cw", "Cm"], "refractive_p45.dat"),
}


def build_coefficients():
    outdir = ROOT / "data" / "coefficients"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, (kfile, cols, nfile) in COEF_SOURCES.items():
        kmat = pd.read_csv(RTM_JL / "data-raw" / kfile, sep=r"\s+", header=None,
                           names=["wavelength", *cols])
        refr = pd.read_csv(RTM_JL / "data-raw" / nfile, sep=r"\s+", header=None,
                           names=["wavelength", "n"])
        df = refr.merge(kmat, on="wavelength")
        assert len(df) == 2101
        df["wavelength"] = df["wavelength"].astype(int)
        n = df["n"].to_numpy()
        df.insert(2, "talf", tav(40, n))
        df.insert(3, "t12", tav(90, n))
        df.insert(4, "t21", df["t12"] / n**2)
        df.to_csv(outdir / f"{name}.csv", index=False, float_format="%.10g")
        print(f"wrote {name}.csv ({len(df)} rows)")


def read_csvy(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def build_validation():
    with open(ROOT / "config" / "validation.json") as f:
        cfg = json.load(f)

    meta = {
        "lopex": read_csvy(SPECTRA_DB / "lopex" / "metadata.csvy"),
        "angers": read_csvy(SPECTRA_DB / "angers" / "metadata.csvy"),
    }
    trait_rows = []
    spectra_rows = []
    for s in cfg["samples"]:
        sid, project = s["id"], s["project"]
        if project == "synthetic":
            continue
        m = meta[project].set_index("observation_id").loc[sid]
        anth = m.get("leaf_anth_per_area", np.nan)
        trait_rows.append({
            "sample_id": sid,
            "project": project,
            "species": m["scientific_name"],
            "role": s["role"],
            "Cab": m["leaf_chltot_per_area"],
            "Car": m["leaf_cartot_per_area"],
            "Canth": np.nan if (pd.isna(anth) or anth < 0) else anth,
            "Cw": m["leaf_water_thickness"],
            "Cm": m["leaf_mass_per_area"],
        })
        spec = read_csvy(SPECTRA_DB / project / "spectra" / f"{sid}.csvy")
        cols = [c for c in spec.columns if c != "wavelengths"]
        # Columns alternate R, T (verified from the csvy spectra_types headers)
        for rep, i in enumerate(range(0, len(cols), 2), start=1):
            spectra_rows.append(pd.DataFrame({
                "sample_id": sid,
                "replicate": rep,
                "wavelength": spec["wavelengths"].astype(int),
                "R": spec[cols[i]],
                "T": spec[cols[i + 1]],
            }))

    # Synthetic spectrum with known parameters
    syn = next(s for s in cfg["samples"] if s["project"] == "synthetic")
    truth = syn["truth"]
    model = Prospect("prospectd")
    R, T = model(truth["N"], truth, e1="exact")
    rng = np.random.default_rng(syn["seed"])
    sd = syn["noise_sd"]
    spectra_rows.append(pd.DataFrame({
        "sample_id": syn["id"],
        "replicate": 1,
        "wavelength": model.wavelength,
        "R": R + rng.normal(0, sd, R.size),
        "T": T + rng.normal(0, sd, T.size),
    }))
    trait_rows.append({
        "sample_id": syn["id"], "project": "synthetic", "species": "synthetic",
        "role": syn["role"], "N": truth["N"],
        **{c: truth[c] for c in ["Cab", "Car", "Canth", "Cbrown", "Cw", "Cm"]},
    })

    traits = pd.DataFrame(trait_rows)
    spectra = pd.concat(spectra_rows, ignore_index=True)
    outdir = ROOT / "data" / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    traits.to_csv(outdir / "traits.csv", index=False)
    spectra.to_csv(outdir / "spectra.csv", index=False, float_format="%.6g")
    print(f"wrote traits.csv ({len(traits)} samples), spectra.csv ({len(spectra)} rows)")
    return traits, spectra


def build_inputs(spectra: pd.DataFrame, versions=("prospectd",)):
    models = json.load(open(ROOT / "config" / "models.json"))
    first = spectra[spectra["replicate"] == 1]
    for version in versions:
        spec = models[version]
        coef = pd.read_csv(ROOT / spec["coefficients"])
        keep = ["wavelength", "talf", "t12", "t21", *spec["constituents"]]
        outdir = ROOT / "data" / "inputs" / version
        outdir.mkdir(parents=True, exist_ok=True)
        for sid, d in first.groupby("sample_id"):
            d = d[np.isfinite(d["R"])][["wavelength", "R", "T"]]
            out = d.merge(coef[keep], on="wavelength").sort_values("wavelength")
            out.to_csv(outdir / f"{sid}.csv", index=False, float_format="%.10g")
        print(f"wrote inputs for {version}: {first['sample_id'].nunique()} samples")


if __name__ == "__main__":
    build_coefficients()
    _, spectra = build_validation()
    build_inputs(spectra)
