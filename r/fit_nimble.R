# Bayesian PROSPECT inversion with NIMBLE (automated-factor slice sampler
# on all parameters jointly). Chains run sequentially.
# Usage: Rscript r/fit_nimble.R --sample lopex_01 [--errmodel iid] [--e1 exact]
source("r/common.R")
source("r/prospect_nimble.R")

args <- parse_args()
inp <- load_inputs(args$sample, args$version)
cfg <- sampling_config("nimble")
if (identical(args$quick, "true")) { cfg$warmup <- 200; cfg$draws <- 200 }
seed <- read_config("sampling")$seed
nc <- length(inp$constituents)

code <- nimbleCode({
  N ~ T(dnorm(muN, sd = sdN), lowerN, )
  for (j in 1:nc) {
    C[j] ~ dlnorm(meanlog[j], sdlog = sdlog[j])
  }
  sigma ~ T(dnorm(0, sd = sigma_scale), 0, )
  mod[1:nwl] <- prospect_refl_nim(N, C[1:nc], K[1:nwl, 1:nc], talf[1:nwl],
                                  t12[1:nwl], t21[1:nwl], e1mode)
  obs[1:nwl] ~ dnorm_iid(mod[1:nwl], sigma)
})

constants <- list(
  nwl = length(inp$obs), nc = nc, K = inp$K,
  talf = inp$talf, t12 = inp$t12, t21 = inp$t21,
  muN = inp$prior_N$mean, sdN = inp$prior_N$sd, lowerN = inp$prior_N$lower,
  meanlog = inp$prior_meanlog, sdlog = inp$prior_sdlog,
  sigma_scale = inp$prior_sigma_scale,
  e1mode = if (args$e1 == "exact") 1L else 2L
)
inits <- make_inits(inp, cfg$chains, seed)

t_setup <- elapsed({
  model <- nimbleModel(code, constants = constants, data = list(obs = inp$obs),
                       inits = inits[[1]], check = FALSE)
  conf <- configureMCMC(model, nodes = NULL, monitors = c("N", "C", "sigma"),
                        print = FALSE)
  conf$addSampler(target = c("N", "C", "sigma"), type = cfg$sampler)
  mcmc <- buildMCMC(conf)
  cmodel <- compileNimble(model)
  cmcmc <- compileNimble(mcmc, project = model)
})

t_sampling <- elapsed(
  samples <- runMCMC(cmcmc, niter = cfg$warmup + cfg$draws, nburnin = cfg$warmup,
                     thin = cfg$thin, nchains = cfg$chains, inits = inits,
                     setSeed = seed + seq_len(cfg$chains), progressBar = FALSE)
)

draws <- rbindlist(lapply(seq_along(samples), function(ch) {
  s <- samples[[ch]]
  d <- data.table(chain = ch, draw = seq_len(nrow(s)), N = s[, "N"])
  for (j in seq_len(nc)) d[[inp$constituents[j]]] <- s[, sprintf("C[%d]", j)]
  d$sigma <- s[, "sigma"]
  d
}))

meta <- base_meta("nimble", args, inp, cfg, t_setup, t_sampling, list(
  divergences = NA,
  software = list(nimble = as.character(packageVersion("nimble")))
))
write_fit(fit_dir("nimble", args), draws, meta)
