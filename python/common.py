"""Shared helpers for the Python backend and summaries (mirrors r/common.R)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_config(name: str):
    with open(ROOT / "config" / f"{name}.json") as f:
        return json.load(f)


def load_inputs(sample_id: str, version: str = "prospectd") -> dict:
    constituents = read_config("models")[version]["constituents"]
    d = pd.read_csv(ROOT / "data" / "inputs" / version / f"{sample_id}.csv")
    priors = read_config("priors")
    cp = [priors["constituents"][c] for c in constituents]
    return {
        "sample_id": sample_id,
        "version": version,
        "constituents": constituents,
        "wavelength": d["wavelength"].to_numpy(),
        "obs": d["R"].to_numpy(),
        "obs_T": d["T"].to_numpy(),
        "K": d[constituents].to_numpy(),
        "talf": d["talf"].to_numpy(),
        "t12": d["t12"].to_numpy(),
        "t21": d["t21"].to_numpy(),
        "prior_meanlog": np.array([c["meanlog"] for c in cp]),
        "prior_sdlog": np.array([c["sdlog"] for c in cp]),
        "prior_N": priors["N"],
        "prior_sigma_scale": priors["sigma"]["scale"],
    }


def make_inits(inp: dict, chains: int, seed: int) -> list[dict]:
    """Shared initial values: prior medians with log-scale jitter (one seed per chain)."""
    ini = read_config("sampling")["init"]
    out = []
    for ch in range(1, chains + 1):
        rng = np.random.default_rng(seed + ch)
        jit = rng.uniform(-ini["jitter"], ini["jitter"], len(inp["prior_meanlog"]))
        out.append({"N": ini["N"], "C": np.exp(inp["prior_meanlog"] + jit),
                    "sigma": ini["sigma"]})
    return out


def fit_dir(backend: str, args) -> Path:
    tag = args.errmodel if args.e1 == "exact" else f"{args.errmodel}-{args.e1}"
    d = ROOT / "results" / "fits" / backend / tag / args.sample
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_fit(d: Path, draws: pd.DataFrame, meta: dict):
    draws.to_csv(d / "draws.csv.gz", index=False)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Wrote {d}")
