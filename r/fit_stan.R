# Bayesian PROSPECT inversion with Stan (cmdstanr). Chains run sequentially.
# Usage: Rscript r/fit_stan.R --sample lopex_01 [--errmodel iid] [--e1 exact]
suppressPackageStartupMessages(library(cmdstanr))
source("r/common.R")

args <- parse_args()
inp <- load_inputs(args$sample, args$version)
cfg <- sampling_config("stan")
seed <- read_config("sampling")$seed

set_cmdstan_path(file.path(Sys.getenv("CONDA_PREFIX"), "bin", "cmdstan"))
dir.create("stan/build", showWarnings = FALSE)
t_compile <- system.time(
  mod <- cmdstan_model(file.path("stan", paste0("prospect_", args$errmodel, ".stan")),
                       include_paths = "stan",
                       user_header = normalizePath("stan/expint.hpp"),
                       dir = "stan/build")
)[["elapsed"]]

nc <- length(inp$constituents)
dat <- list(
  nwl = length(inp$obs), nc = nc, obs = inp$obs, K = inp$K,
  talf = inp$talf, t12 = inp$t12, t21 = inp$t21,
  wavelength = inp$wavelength,
  prior_meanlog = as.array(inp$prior_meanlog), prior_sdlog = as.array(inp$prior_sdlog),
  prior_N_mean = inp$prior_N$mean, prior_N_sd = inp$prior_N$sd,
  prior_N_lower = inp$prior_N$lower, prior_sigma_scale = inp$prior_sigma_scale,
  e1mode = if (args$e1 == "exact") 1L else 2L
)
inits <- make_inits(inp, cfg$chains, seed)
if (args$errmodel != "iid") {
  # extra error-model parameters start at neutral values
  inits <- lapply(inits, function(x) {
    x$sigma <- NULL
    c(x, list(sigma0 = 0.005, sigma1 = 0.01, phi = 0.5))
  })
}

t_sampling <- elapsed(
  fit <- mod$sample(
    data = dat, init = inits, seed = seed, chains = cfg$chains, parallel_chains = 1,
    iter_warmup = cfg$warmup, iter_sampling = cfg$draws,
    adapt_delta = cfg$adapt_delta, refresh = 0, show_messages = FALSE
  )
)

vars <- setdiff(fit$metadata()$stan_variables, "lp__")
dr <- posterior::as_draws_df(fit$draws(vars))
draws <- data.frame(chain = dr$.chain, draw = dr$.iteration, N = dr$N)
for (j in seq_len(nc)) draws[[inp$constituents[j]]] <- dr[[sprintf("C[%d]", j)]]
for (v in setdiff(names(dr), c("N", sprintf("C[%d]", seq_len(nc)), ".chain", ".iteration", ".draw"))) {
  draws[[v]] <- dr[[v]]
}

diag <- fit$diagnostic_summary(quiet = TRUE)
meta <- base_meta("stan", args, inp, cfg, t_compile, t_sampling, list(
  divergences = sum(diag$num_divergent),
  max_treedepth_hits = sum(diag$num_max_treedepth),
  time_per_chain_s = fit$time()$chains$total,
  software = list(cmdstan = cmdstan_version(), cmdstanr = as.character(packageVersion("cmdstanr")))
))
write_fit(fit_dir("stan", args), draws, meta)
