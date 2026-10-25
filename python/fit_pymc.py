"""Bayesian PROSPECT inversion with a PyMC model and one of three NUTS samplers.
Chains run sequentially on one core.

  --sampler pymc     PyMC's NUTS (Python), log-density + gradient compiled by
                     PyTensor's default numba backend          -> backend "pymc"
  --sampler nutpie   nutpie (Rust NUTS), same numba-compiled
                     log-density + gradient                     -> "pymc-nutpie"
  --sampler numpyro  NumPyro NUTS, log-density compiled by PyTensor's JAX
                     backend, gradient by JAX, whole sampler jit-compiled
                     (CPU)                                      -> "pymc-numpyro"

  --e1 exact|fast|approx   (see python/prospect_pt.py)

Usage: python python/fit_pymc.py --sample lopex_01 [--sampler pymc] [--e1 exact]
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

# One core per fit, for every library (set before numpy/jax import)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

warnings.filterwarnings("ignore", message="Cannot cache compiled function")
warnings.filterwarnings("ignore", category=FutureWarning)

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


def _steps(ds):
    return int(ds["n_steps"].values.sum()) if ds is not None and "n_steps" in ds else 0


def run_pymc(model, inits, cfg, seed):
    t0 = time.perf_counter()
    idata = pm.sample(
        draws=cfg["draws"], tune=cfg["warmup"], chains=cfg["chains"], cores=1,
        initvals=inits, init="adapt_diag", target_accept=cfg["target_accept"],
        random_seed=seed, progressbar=False, compute_convergence_checks=False,
        discard_tuned_samples=False, model=model,
    )
    t_total = time.perf_counter() - t0
    t_sampling = float(idata.posterior.attrs["sampling_time"])
    post = {"N": idata.posterior["N"].values, "C": idata.posterior["C"].values,
            "sigma": idata.posterior["sigma"].values}
    ss = idata.sample_stats
    ws = idata.warmup_sample_stats if "warmup_sample_stats" in idata else None
    stats = {"n_grad_evals": _steps(ss) + _steps(ws),
             "divergences": int(ss["diverging"].values.sum()),
             "max_treedepth_hits": int(ss["reached_max_treedepth"].values.sum())
             if "reached_max_treedepth" in ss else None}
    return post, t_total - t_sampling, t_sampling, stats, {"pymc": pm.__version__}


def run_nutpie(model, inp, cfg, seed):
    import nutpie

    ini = read_config("sampling")["init"]
    t0 = time.perf_counter()
    # nutpie takes one initial point (the shared prior-median start), jittered per chain
    compiled = nutpie.compile_pymc_model(
        model, backend="numba",
        initial_points={"N": ini["N"], "C": np.exp(inp["prior_meanlog"]), "sigma": ini["sigma"]})
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    tr = nutpie.sample(compiled, draws=cfg["draws"], tune=cfg["warmup"], chains=cfg["chains"],
                       cores=1, seed=seed, save_warmup=True, progress_bar=False,
                       target_accept=cfg["target_accept"])
    t_sampling = time.perf_counter() - t0
    post = {"N": tr.posterior["N"].values, "C": tr.posterior["C"].values,
            "sigma": tr.posterior["sigma"].values}
    ss = tr.sample_stats
    ws = tr.warmup_sample_stats if "warmup_sample_stats" in tr else None
    stats = {"n_grad_evals": _steps(ss) + _steps(ws),
             "divergences": int(ss["diverging"].values.sum()),
             "max_treedepth_hits": int(ss["maxdepth_reached"].values.sum())
             if "maxdepth_reached" in ss else None}
    return post, t_setup, t_sampling, stats, {"nutpie": nutpie.__version__}


def run_numpyro(model, inits, cfg, seed):
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from pymc.initial_point import make_initial_point_fn
    from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp

    t0 = time.perf_counter()
    # NB: despite the flag name, negative_logp=False returns the *potential*
    # (-log p), which is what PyMC itself passes to NumPyro as potential_fn.
    potential = get_jaxified_logp(model, negative_logp=False)
    value_vars = model.value_vars
    # map transformed (unconstrained) values -> constrained N, C, sigma
    out_vars = [v for v in model.unobserved_value_vars if v.name in ("N", "C", "sigma")]
    _graph = get_jaxified_graph(inputs=value_vars, outputs=out_vars)
    to_constrained = jax.jit(jax.vmap(lambda xs: _graph(*xs)))
    t_build = time.perf_counter() - t0

    kernel = NUTS(potential_fn=potential, target_accept_prob=cfg["target_accept"])
    mcmc = MCMC(kernel, num_warmup=cfg["warmup"], num_samples=cfg["draws"], num_chains=1,
                progress_bar=False)
    key = jax.random.PRNGKey(seed)
    chain_times, samples, n_steps, divs = [], [], 0, 0
    for ch, iv in enumerate(inits):
        p0 = make_initial_point_fn(model=model, overrides=iv, jitter_rvs=set())(0)
        init = [np.asarray(p0[v.name], dtype=float) for v in value_vars]
        key, k1 = jax.random.split(key)
        tc = time.perf_counter()
        mcmc.warmup(k1, init_params=init, collect_warmup=True, extra_fields=("num_steps",))
        n_steps += int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps", "diverging"))
        ef = mcmc.get_extra_fields()
        x = mcmc.get_samples()
        jax.block_until_ready(x)
        chain_times.append(time.perf_counter() - tc)
        n_steps += int(np.asarray(ef["num_steps"]).sum())
        divs += int(np.asarray(ef["diverging"]).sum())
        samples.append([np.asarray(a) for a in to_constrained(x)])
    # JIT compilation happens in the first chain: setup = excess time of chain 1
    t_compile = max(chain_times[0] - float(np.median(chain_times[1:])), 0.0)
    t_sampling = sum(chain_times) - t_compile
    names = [v.name for v in out_vars]
    post = {n: np.stack([s[i] for s in samples]) for i, n in enumerate(names)}
    stats = {"n_grad_evals": n_steps, "divergences": divs, "max_treedepth_hits": None}
    return post, t_build + t_compile, t_sampling, stats, {"numpyro": numpyro.__version__,
                                                          "jax": jax.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="synthetic")
    ap.add_argument("--errmodel", default="iid", choices=["iid"])
    ap.add_argument("--e1", default="exact", choices=["exact", "fast", "approx"])
    ap.add_argument("--sampler", default="pymc", choices=["pymc", "nutpie", "numpyro"])
    ap.add_argument("--version", default="prospectd")
    args = ap.parse_args()

    inp = load_inputs(args.sample, args.version)
    cfg = read_config("sampling")["pymc"]
    seed = read_config("sampling")["seed"]
    model = build_model(inp, args.e1)
    inits = make_inits(inp, cfg["chains"], seed)

    if args.sampler == "pymc":
        post, t_setup, t_sampling, stats, sw = run_pymc(model, inits, cfg, seed)
    elif args.sampler == "nutpie":
        post, t_setup, t_sampling, stats, sw = run_nutpie(model, inp, cfg, seed)
    else:
        post, t_setup, t_sampling, stats, sw = run_numpyro(model, inits, cfg, seed)

    nchain, ndraw = post["N"].shape[:2]
    draws = pd.DataFrame({
        "chain": np.repeat(np.arange(1, nchain + 1), ndraw),
        "draw": np.tile(np.arange(1, ndraw + 1), nchain),
        "N": post["N"].ravel(),
    })
    Cv = post["C"].reshape(nchain * ndraw, -1)
    for j, c in enumerate(inp["constituents"]):
        draws[c] = Cv[:, j]
    draws["sigma"] = post["sigma"].ravel()

    backend = "pymc" if args.sampler == "pymc" else f"pymc-{args.sampler}"
    meta = {
        "backend": backend,
        "sampler": args.sampler,
        "sample_id": args.sample,
        "errmodel": args.errmodel,
        "e1": args.e1,
        "version": args.version,
        "n_wavelengths": len(inp["obs"]),
        "constituents": inp["constituents"],
        "settings": cfg,
        "time_setup_s": t_setup,
        "time_sampling_s": t_sampling,
        **stats,
        "time_per_grad_us": 1e6 * t_sampling / stats["n_grad_evals"] if stats["n_grad_evals"] else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": f"Python {platform.python_version()}",
        "software": {**sw, "pymc": pm.__version__, "pytensor": pytensor.__version__},
    }
    write_fit(fit_dir(backend, args), draws, meta)


if __name__ == "__main__":
    main()
