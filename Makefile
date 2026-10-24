# Reproduce the full comparison:  make -j4 all
#
# Toolchain commands can be overridden, e.g.
#   make PY=python3 RSCRIPT=Rscript JULIA="julia --project=julia"

PY      ?= /opt/pyenv/bin/python
RSCRIPT ?= /opt/mamba/bin/micromamba run -r /opt/mamba -n rstan Rscript
JULIA   ?= env JULIA_PKG_SERVER= JULIA_DEPOT_PATH=/opt/julia-depot /opt/mamba/envs/jl/bin/julia --project=julia
QUARTO  ?= /opt/mamba/bin/micromamba run -r /opt/mamba -n jl quarto

SAMPLES  := $(basename $(notdir $(wildcard data/inputs/prospectd/*.csv)))
BACKENDS := stan pymc turing nimble bayestools

# Main comparison: every backend, iid error model, exact E1
IID_FITS    := $(foreach b,$(BACKENDS),$(foreach s,$(SAMPLES),results/fits/$(b)/iid/$(s)/meta.json))
# Sensitivity to the E1 approximation (Stan)
APPROX_FITS := $(foreach s,$(SAMPLES),results/fits/stan/iid-approx/$(s)/meta.json)
# Error-model demonstration (Stan)
HAR1_FITS   := $(foreach s,$(SAMPLES),results/fits/stan/hetero_ar1/$(s)/meta.json)

LOG = results/logs/$(subst /,_,$(patsubst results/fits/%/meta.json,%,$@)).log

.PHONY: all data forward fits summary site stan-compile clean-fits
all: forward fits summary site

data:
	$(PY) scripts/prepare_data.py

# ---- Forward-model verification -------------------------------------------
forward:
	mkdir -p tests/forward/out
	$(PY) tests/forward/run_python.py
	$(RSCRIPT) tests/forward/run_r.R
	$(JULIA) tests/forward/run_julia.jl
	$(PY) tests/forward/compare_forward.py

# ---- Fits ------------------------------------------------------------------
fits: $(IID_FITS) $(APPROX_FITS) $(HAR1_FITS)

# Compile Stan models once, before parallel fits use them
stan/build/.compiled: stan/*.stan stan/expint.hpp
	mkdir -p stan/build results/logs
	$(RSCRIPT) stan/compile_models.R
	touch $@

results/fits/stan/iid/%/meta.json: data/inputs/prospectd/%.csv | stan/build/.compiled
	@mkdir -p results/logs
	$(RSCRIPT) r/fit_stan.R --sample $* > $(LOG) 2>&1

results/fits/stan/iid-approx/%/meta.json: data/inputs/prospectd/%.csv | stan/build/.compiled
	@mkdir -p results/logs
	$(RSCRIPT) r/fit_stan.R --sample $* --e1 approx > $(LOG) 2>&1

results/fits/stan/hetero_ar1/%/meta.json: data/inputs/prospectd/%.csv | stan/build/.compiled
	@mkdir -p results/logs
	$(RSCRIPT) r/fit_stan.R --sample $* --errmodel hetero_ar1 > $(LOG) 2>&1

results/fits/pymc/iid/%/meta.json: data/inputs/prospectd/%.csv
	@mkdir -p results/logs
	$(PY) python/fit_pymc.py --sample $* > $(LOG) 2>&1

results/fits/turing/iid/%/meta.json: data/inputs/prospectd/%.csv
	@mkdir -p results/logs
	$(JULIA) julia/fit_turing.jl --sample $* > $(LOG) 2>&1

results/fits/nimble/iid/%/meta.json: data/inputs/prospectd/%.csv
	@mkdir -p results/logs
	$(RSCRIPT) r/fit_nimble.R --sample $* > $(LOG) 2>&1

results/fits/bayestools/iid/%/meta.json: data/inputs/prospectd/%.csv
	@mkdir -p results/logs
	$(RSCRIPT) r/fit_bayestools.R --sample $* > $(LOG) 2>&1

# ---- Summaries and website ---------------------------------------------------
summary:
	$(PY) python/summarize.py

site:
	cd site && QUARTO_PYTHON=$(PY) $(QUARTO) render

clean-fits:
	rm -rf results/fits results/logs
