# Bayesian PROSPECT inversion with BayesianTools (DEzs sampler) and the
# vectorized base-R PROSPECT. Each BayesianTools "chain" is a DEzs run with
# 3 internal chains; every internal chain is reported as its own chain.
# Runs execute sequentially.
# Usage: Rscript r/fit_bayestools.R --sample lopex_01 [--errmodel iid] [--e1 exact]
suppressPackageStartupMessages(library(BayesianTools))
source("r/common.R")
source("r/prospect.R")

args <- parse_args()
inp <- load_inputs(args$sample, args$version)
cfg <- sampling_config("bayestools")
if (identical(args$quick, "true")) { cfg$iterations <- 3000; cfg$burnin <- 1500 }
seed <- read_config("sampling")$seed
nc <- length(inp$constituents)
pnames <- c("N", inp$constituents, "sigma")
pN <- inp$prior_N

loglik <- function(par) {
  R <- prospect(par[1], par[2:(nc + 1)], inp, e1 = args$e1)$R
  if (!all(is.finite(R))) return(-Inf)
  sum(dnorm(inp$obs, R, par[nc + 2], log = TRUE))
}

prior_density <- function(par) {
  N <- par[1]; C <- par[2:(nc + 1)]; sigma <- par[nc + 2]
  if (N <= pN$lower || any(C <= 0) || sigma <= 0) return(-Inf)
  dnorm(N, pN$mean, pN$sd, log = TRUE) -
    pnorm(pN$lower, pN$mean, pN$sd, lower.tail = FALSE, log.p = TRUE) +
    sum(dlnorm(C, inp$prior_meanlog, inp$prior_sdlog, log = TRUE)) +
    dnorm(sigma, 0, inp$prior_sigma_scale, log = TRUE) + log(2)
}

prior_sampler <- function(n = 1) {
  u <- runif(n, pnorm(pN$lower, pN$mean, pN$sd), 1)
  N <- qnorm(u, pN$mean, pN$sd)
  C <- vapply(seq_len(nc), function(j) rlnorm(n, inp$prior_meanlog[j], inp$prior_sdlog[j]),
              numeric(n))
  sigma <- abs(rnorm(n, 0, inp$prior_sigma_scale))
  cbind(N, matrix(C, nrow = n), sigma)
}

prior <- createPrior(density = prior_density, sampler = prior_sampler,
                     lower = c(pN$lower, rep(0, nc), 0),
                     upper = c(Inf, rep(Inf, nc), Inf))
setup <- createBayesianSetup(loglik, prior, names = pnames, parallel = FALSE)

# Shared initial values: each run's 3 internal chains start from 3 jittered
# prior-median points.
inits <- make_inits(inp, cfg$chains * 3, seed)
start_mat <- function(run) {
  do.call(rbind, lapply(inits[(run - 1) * 3 + 1:3], function(x) c(x$N, x$C, x$sigma)))
}

t_sampling <- 0
runs <- vector("list", cfg$chains)
for (run in seq_len(cfg$chains)) {
  set.seed(seed + run)
  t_sampling <- t_sampling + elapsed(
    runs[[run]] <- runMCMC(setup, sampler = cfg$sampler, settings = list(
      iterations = cfg$iterations, burnin = 0, startValue = start_mat(run),
      message = FALSE
    ))
  )
}

# Each DEzs run stores iterations/3 rows per internal chain
draws <- rbindlist(lapply(seq_along(runs), function(run) {
  ch <- runs[[run]]$chain
  rbindlist(lapply(seq_along(ch), function(i) {
    m <- as.matrix(ch[[i]])[, seq_along(pnames), drop = FALSE]
    keep <- seq(cfg$burnin / 3 + 1, nrow(m), by = cfg$thin)
    d <- as.data.table(m[keep, , drop = FALSE])
    setnames(d, pnames)
    cbind(data.table(chain = (run - 1) * 3 + i, draw = seq_along(keep)), d)
  }))
}))

meta <- base_meta("bayestools", args, inp, cfg, 0, t_sampling, list(
  divergences = NA,
  n_internal_chains = 3 * cfg$chains,
  software = list(BayesianTools = as.character(packageVersion("BayesianTools")))
))
write_fit(fit_dir("bayestools", args), draws, meta)
