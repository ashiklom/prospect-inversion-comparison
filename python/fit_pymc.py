"""Bayesian PROSPECT inversion with PyMC (NUTS). Chains run sequentially.

Usage: python python/fit_pymc.py --sample lopex_01 [--errmodel iid] [--e1 exact]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

warnings.filterwarnings("ignore", message="Cannot cache compiled function")

import pymc as pm  # noqa: E402
import pytensor  # noqa: E402

import prospect_pt  # noqa: E402
from common import fit_dir, load_inputs, make_inits, read_config, write_fit  # noqa: E402


def build_model(inp, e1):
    pN = inp["prior_N"]
    with pm.Model() as model:
        N = pm.TruncatedNormal("N", mu=pN["mean"], sigma=pN["sd"], lower=pN["lower"])
        C = pm.LogNormal("C", mu=inp["prior_meanlog"], sigma=inp["prior_sdlog"],
                         shape=len(inp["constituents"]))
        sigma = pm.HalfNormal("sigma", sigma=inp["prior_sigma_scale"])
        R = prospect_pt.prospect(N, C, inp["K"], inp["talf"], inp["t12"], inp["t21"], e1=e1)
        pm.Normal("obs", mu=R, sigma=sigma, observed=inp["obs"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="synthetic")
    ap.add_argument("--errmodel", default="iid", choices=["iid"])
    ap.add_argument("--e1", default="exact", choices=["exact", "approx"])
    ap.add_argument("--version", default="prospectd")
    args = ap.parse_args()

    inp = load_inputs(args.sample, args.version)
    cfg = read_config("sampling")["pymc"]
    seed = read_config("sampling")["seed"]

    t0 = time.perf_counter()
    model = build_model(inp, args.e1)
    initvals = make_inits(inp, cfg["chains"], seed)
    idata = pm.sample(
        draws=cfg["draws"], tune=cfg["warmup"], chains=cfg["chains"], cores=1,
        initvals=initvals, init="adapt_diag", target_accept=cfg["target_accept"],
        random_seed=seed, progressbar=False, compute_convergence_checks=False,
        model=model,
    )
    t_total = time.perf_counter() - t0
    t_sampling = float(idata.posterior.attrs["sampling_time"])

    post = idata.posterior
    nchain, ndraw = post.sizes["chain"], post.sizes["draw"]
    draws = pd.DataFrame({
        "chain": np.repeat(np.arange(1, nchain + 1), ndraw),
        "draw": np.tile(np.arange(1, ndraw + 1), nchain),
        "N": post["N"].values.ravel(),
    })
    Cv = post["C"].values.reshape(nchain * ndraw, -1)
    for j, c in enumerate(inp["constituents"]):
        draws[c] = Cv[:, j]
    draws["sigma"] = post["sigma"].values.ravel()

    divergences = int(idata.sample_stats["diverging"].values.sum())
    meta = {
        "backend": "pymc",
        "sample_id": args.sample,
        "errmodel": args.errmodel,
        "e1": args.e1,
        "version": args.version,
        "n_wavelengths": len(inp["obs"]),
        "constituents": inp["constituents"],
        "settings": cfg,
        "time_setup_s": t_total - t_sampling,
        "time_sampling_s": t_sampling,
        "divergences": divergences,
        "max_treedepth_hits": int(idata.sample_stats["reached_max_treedepth"].values.sum())
        if "reached_max_treedepth" in idata.sample_stats else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"Python {platform.python_version()}",
        "software": {"pymc": pm.__version__, "pytensor": pytensor.__version__,
                     "pytensor_linker": str(pytensor.config.linker)},
    }
    write_fit(fit_dir("pymc", args), draws, meta)


if __name__ == "__main__":
    main()
