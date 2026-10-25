# PROSPECT inversion comparison

Bayesian inversion of the **PROSPECT-D** leaf radiative transfer model,
implemented separately in five toolchains and compared on real leaves with
measured traits:

| Backend | Language | PROSPECT implementation | Sampler |
|---|---|---|---|
| BayesianTools | R | `r/prospect.R` (vectorized base R) | DEzs |
| NIMBLE | R / NIMBLE DSL | `r/prospect_nimble.R` (`nimbleFunction`) | automated-factor slice |
| Stan | Stan (via cmdstanr) | `stan/prospect_functions.stan` | NUTS |
| Turing.jl | Julia | `julia/src/ProspectModels.jl` | NUTS (ForwardDiff) |
| PyMC | Python / PyTensor | `python/prospect_pt.py` | NUTS |

`python/prospect_np.py` is the NumPy reference. Every implementation is
verified against it to about 1e-15 (`make forward`), and the reference is
checked against the independent [`prosail`](https://github.com/jgomezdans/prosail)
package.

**Results website:** rendered into [`docs/`](docs/index.html). Enable GitHub
Pages from the `docs/` folder on `main` to publish it.

## Design

* **Version-agnostic forward model.** A PROSPECT version is a coefficient
  table plus a list of constituents (`config/models.json`,
  `data/coefficients/*.csv`). PROSPECT-D, 5B, 5 and 4 are defined. Adding PRO
  means adding its coefficient table, with no backend code changes.
* **Exponential integral E1.** Each backend has an `exact` mode and an
  `approx` mode (Swamee–Ohija closed form). In `exact` mode it calls a library
  E1 and supplies the analytic derivative −e⁻ˣ/x explicitly:
  * Boost via an external C++ header with `make_callback_var` (Stan)
  * a custom PyTensor `Op` with `pullback`, plus a native numba kernel (PyMC)
  * `SpecialFunctions.expint` with its DiffRules rule (Julia)
  * GSL via `nimbleExternalCall` (NIMBLE)
  * `expint::expint_E1` (R)
* **Identical statistical model everywhere.** Priors are in
  `config/priors.json`, sampler settings and initial values in
  `config/sampling.json`. Chains run sequentially on one core in every backend,
  so wall times are comparable.
* **Validation.** The fits use reflectance only. Measured chlorophyll,
  carotenoids, EWT and LMA, plus the **measured transmittance**, are held out
  for validation.
* **Error models.** The first round uses iid normal residuals.
  `stan/prospect_hetero_ar1.stan` demonstrates a heteroskedastic AR(1)
  likelihood (exact, O(n)). The website's *Error models* page discusses more
  complete options.

## Layout

```
config/            models, priors, sampling settings, validation sample list
data/coefficients/ PROSPECT coefficient tables (+ precomputed interface transmissivities)
data/validation/   traits.csv, spectra.csv (all LOPEX replicates)
data/inputs/       fit-ready per-sample tables (observed R/T + coefficients)
python/ r/ stan/ julia/   forward models and fit scripts per backend
tests/forward/     cross-language forward-model verification
results/fits/      <backend>/<errmodel>/<sample>/{draws.csv.gz, meta.json}
results/summary/   tables produced by python/summarize.py (drive the website)
site/              Quarto website source -> docs/
scripts/prepare_data.py   builds data/ from ../RTM.jl and ../spectra_db
```

## Reproducing

Toolchains (conda-forge via micromamba, plus a Python venv):

```bash
micromamba env create -f environment-r.yml        # R, NIMBLE, cmdstanr + CmdStan, GSL
git clone https://github.com/florianhartig/BayesianTools && \
  micromamba run -n rstan R CMD INSTALL BayesianTools/BayesianTools
micromamba env create -f environment-julia.yml    # Julia + Quarto
micromamba run -n jl julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python -m venv .venv && .venv/bin/pip install -r python/requirements.txt
```

Then (override the tool paths at the top of the `Makefile` as needed):

```bash
make data         # only needed if ../RTM.jl or ../spectra_db change
make forward      # verify all forward models
make -j4 fits     # all fits (about 2 h on 4 cores)
make summary
make site
```

Single fits:

```bash
Rscript r/fit_stan.R --sample lopex_01 [--errmodel iid|hetero_ar1] [--e1 exact|approx]
Rscript r/fit_nimble.R --sample lopex_01
Rscript r/fit_bayestools.R --sample lopex_01
julia --project=julia julia/fit_turing.jl --sample lopex_01
python python/fit_pymc.py --sample lopex_01 [--sampler pymc|nutpie|numpyro] [--e1 exact|fast|approx]
python python/bench_logp.py     # log-density + gradient microbenchmarks (numba vs JAX)
```

The PyMC model can be sampled by `pm.sample`, nutpie (Rust NUTS on the same
numba-compiled gradient) or NumPyro (JAX), with exact, fast (NAG Chebyshev) or
Swamee–Ohija E1. `make pyfits` runs the full 3 × 3 matrix; results are on the
website's *Faster PyMC* page.

The earlier exploratory DynamicHMC, Turing and Stan scripts are preserved on
the `legacy-experiments` branch.
