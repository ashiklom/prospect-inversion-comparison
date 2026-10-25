"""Microbenchmarks for the PyMC model: E1 alone and log-density + gradient,
PyTensor's numba backend (used by pm.sample and nutpie) vs. its JAX backend
(used by NumPyro), for each E1 mode. Single-threaded.

Writes results/summary/bench_logp.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
warnings.filterwarnings("ignore")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
from pymc.initial_point import make_initial_point_fn  # noqa: E402
from pymc.sampling.jax import get_jaxified_logp  # noqa: E402

import prospect_pt  # noqa: E402
from common import load_inputs, make_inits  # noqa: E402
from fit_pymc import build_model  # noqa: E402

SAMPLE = "lopex_01"
REPS = 7  # report the median of REPS timing blocks


def timeit(fn, n):
    fn()
    blocks = []
    for _ in range(REPS):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        blocks.append((time.perf_counter() - t) / n * 1e6)
    return float(np.median(blocks))


def main():
    inp = load_inputs(SAMPLE)
    k = inp["K"] @ np.exp(inp["prior_meanlog"]) / 1.5  # realistic absorption coefficients
    rows = []
    for e1 in ("exact", "fast", "approx"):
        # E1 alone
        x = pt.vector("x")
        f_nb = pytensor.function([x], prospect_pt.E1_MODES[e1](x))
        f_jx = pytensor.function([x], prospect_pt.E1_MODES[e1](x), mode="JAX")
        rows.append({"what": "E1 only", "framework": "numba", "e1": e1, "us": timeit(lambda: f_nb(k), 2000)})
        rows.append({"what": "E1 only", "framework": "JAX",
                     "e1": e1, "us": timeit(lambda: np.asarray(f_jx(k)), 2000)})

        # log-density + gradient of the full model
        model = build_model(inp, e1)
        f = model.logp_dlogp_function(ravel_inputs=True)
        f.set_extra_values({})
        p0 = make_initial_point_fn(model=model, overrides=make_inits(inp, 1, 1)[0], jitter_rvs=set())(0)
        x0 = np.concatenate([np.atleast_1d(p0[v.name]).ravel() for v in f._grad_vars])
        rows.append({"what": "log-density + gradient", "framework": "numba", "e1": e1,
                     "us": timeit(lambda: f(x0), 1000)})
        pot = get_jaxified_logp(model, negative_logp=False)
        xs = [np.asarray(p0[v.name], float) for v in model.value_vars]
        vg = jax.jit(jax.value_and_grad(pot))
        rows.append({"what": "log-density + gradient", "framework": "JAX", "e1": e1,
                     "us": timeit(lambda: jax.block_until_ready(vg(xs)), 300 if e1 == "exact" else 1000)})
        print(rows[-4:])
    out = ROOT / "results" / "summary" / "bench_logp.csv"
    df = pd.DataFrame(rows)
    df["n_wavelengths"] = len(inp["obs"])
    df["sample_id"] = SAMPLE
    df["software"] = f"pytensor {pytensor.__version__}; jax {jax.__version__}"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
