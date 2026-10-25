"""Summarize all fits into tables used by the website.

Reads   results/fits/<backend>/<errtag>/<sample>/{draws.csv.gz, meta.json}
Writes  results/summary/
  fits.csv         one row per fit: timing, divergences, worst R-hat / ESS, ESS/s
  params.csv       per-parameter posterior summaries + measured value + coverage
  ppc.csv.gz       per-wavelength posterior (predictive) quantiles for R and
                   held-out T, plus observations
  validation.csv   R (in-sample) and T (held-out) RMSE, bias and 90% coverage
                   by spectral region
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from common import load_inputs  # noqa: E402
from prospect_np import gpm  # noqa: E402

FITS = ROOT / "results" / "fits"
OUT = ROOT / "results" / "summary"
N_PPC_DRAWS = 400
REGIONS = [("VIS", 400, 700), ("Red edge", 700, 750), ("NIR", 750, 1300),
           ("SWIR1", 1300, 1900), ("SWIR2", 1900, 2500)]
TRAIT_PARAMS = ["Cab", "Car", "Canth", "Cbrown", "Cw", "Cm", "N"]

warnings.filterwarnings("ignore", category=RuntimeWarning)


def chain_array(d: pd.DataFrame, col: str) -> np.ndarray:
    """(chain, draw) array, truncating chains to equal length."""
    groups = [g[col].to_numpy() for _, g in d.groupby("chain")]
    n = min(len(g) for g in groups)
    return np.stack([g[:n] for g in groups])


def model_rt(draws: pd.DataFrame, inp: dict, e1: str):
    C = draws[inp["constituents"]].to_numpy()
    N = draws["N"].to_numpy()
    k = (C @ inp["K"].T) / N[:, None]
    return gpm(k, N[:, None], inp["talf"], inp["t12"], inp["t21"], e1=e1)


def region_of(wl):
    out = np.full(wl.shape, "", dtype=object)
    for name, lo, hi in REGIONS:
        out[(wl >= lo) & (wl < hi)] = name
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traits = pd.read_csv(ROOT / "data" / "validation" / "traits.csv").set_index("sample_id")
    fit_rows, param_rows, ppc_frames, val_rows = [], [], [], []

    for meta_path in sorted(FITS.glob("*/*/*/meta.json")):
        fdir = meta_path.parent
        backend, errtag, sample = fdir.parts[-3:]
        meta = json.load(open(meta_path))
        d = pd.read_csv(fdir / "draws.csv.gz")
        inp = load_inputs(sample, meta.get("version", "prospectd"))
        params = [c for c in d.columns if c not in ("chain", "draw")]
        n_chains = d["chain"].nunique()

        ess_b, ess_t, rhats = {}, {}, {}
        for p in params:
            arr = chain_array(d, p)
            ess_b[p] = float(az.ess(arr, method="bulk"))
            ess_t[p] = float(az.ess(arr, method="tail", prob=(0.05, 0.95)))
            rhats[p] = float(az.rhat(arr, method="rank"))
            q = np.quantile(d[p], [0.05, 0.25, 0.5, 0.75, 0.95])
            measured = traits.loc[sample, p] if (p in traits.columns and sample in traits.index) else np.nan
            param_rows.append({
                "backend": backend, "errmodel": errtag, "sample_id": sample, "param": p,
                "mean": d[p].mean(), "sd": d[p].std(),
                "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4],
                "ess_bulk": ess_b[p], "ess_tail": ess_t[p], "rhat": rhats[p],
                "measured": measured,
                "covered90": bool(q[0] <= measured <= q[4]) if np.isfinite(measured) else None,
            })

        core = ["N", *inp["constituents"]]
        t_samp = meta["time_sampling_s"]
        fit_rows.append({
            "backend": backend, "errmodel": errtag, "sample_id": sample,
            "e1": meta.get("e1"), "n_chains": n_chains, "n_draws": len(d),
            "time_setup_s": meta.get("time_setup_s"), "time_sampling_s": t_samp,
            "divergences": meta.get("divergences"),
            "mean_leapfrog": meta.get("mean_leapfrog"),
            "median_stepsize": meta.get("median_stepsize"),
            "max_rhat": max(rhats[p] for p in params),
            "min_ess_bulk": min(ess_b[p] for p in params),
            "min_ess_tail": min(ess_t[p] for p in params),
            "median_ess_bulk": float(np.median([ess_b[p] for p in core])),
            "min_ess_bulk_per_s": min(ess_b[p] for p in params) / t_samp,
            "median_ess_bulk_per_s": float(np.median([ess_b[p] for p in core])) / t_samp,
        })

        # Posterior predictive (reflectance, fitted) and held-out transmittance
        rng = np.random.default_rng(0)
        sub = d.iloc[rng.choice(len(d), min(N_PPC_DRAWS, len(d)), replace=False)]
        R, T = model_rt(sub, inp, meta.get("e1", "exact"))
        sig = sub["sigma"].to_numpy()[:, None] if "sigma" in sub else 0.0
        if "sigma0" in sub:  # heteroskedastic: sd = sigma0 + sigma1 * R
            sig = sub["sigma0"].to_numpy()[:, None] + sub["sigma1"].to_numpy()[:, None] * R
        Rp = R + rng.normal(size=R.shape) * sig
        Tp = T + rng.normal(size=T.shape) * sig
        qR = np.quantile(R, [0.05, 0.5, 0.95], axis=0)
        qRp = np.quantile(Rp, [0.05, 0.95], axis=0)
        qT = np.quantile(T, [0.05, 0.5, 0.95], axis=0)
        qTp = np.quantile(Tp, [0.05, 0.95], axis=0)
        wl = inp["wavelength"]
        ppc = pd.DataFrame({
            "backend": backend, "errmodel": errtag, "sample_id": sample, "wavelength": wl,
            "R_obs": inp["obs"], "R_q05": qR[0], "R_q50": qR[1], "R_q95": qR[2],
            "R_pred_q05": qRp[0], "R_pred_q95": qRp[1],
            "T_obs": inp["obs_T"], "T_q05": qT[0], "T_q50": qT[1], "T_q95": qT[2],
            "T_pred_q05": qTp[0], "T_pred_q95": qTp[1],
        })
        ppc_frames.append(ppc)

        reg = region_of(wl)
        for quantity in ("R", "T"):
            obs = ppc[f"{quantity}_obs"].to_numpy()
            med = ppc[f"{quantity}_q50"].to_numpy()
            lo, hi = ppc[f"{quantity}_pred_q05"].to_numpy(), ppc[f"{quantity}_pred_q95"].to_numpy()
            for name in ["All", *[r[0] for r in REGIONS]]:
                m = np.isfinite(obs) & ((reg == name) if name != "All" else True)
                if m.sum() == 0:
                    continue
                err = med[m] - obs[m]
                val_rows.append({
                    "backend": backend, "errmodel": errtag, "sample_id": sample,
                    "quantity": quantity, "region": name, "n": int(m.sum()),
                    "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(np.mean(err)),
                    "coverage90": float(np.mean((obs[m] >= lo[m]) & (obs[m] <= hi[m]))),
                })
        print(f"summarized {backend}/{errtag}/{sample}")

    pd.DataFrame(fit_rows).to_csv(OUT / "fits.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "params.csv", index=False)
    pd.concat(ppc_frames).to_csv(OUT / "ppc.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(val_rows).to_csv(OUT / "validation.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
