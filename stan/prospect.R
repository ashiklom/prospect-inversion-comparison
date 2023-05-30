## pak::pak("https://github.com/stan-dev/cmdstanr")
library(cmdstanr)
library(posterior)
## install_cmdstan()
## library(rstan)
## options(mc.cores = 1)
# set_cmdstan_path("~/src/cmdstan/")

stanfile <- "./prospect.stan"

obs_raw <- read.csv("~/projects/spectra_db/lopex/spectra/lopex_01.csvy", comment.char = "#")

obs <- obs_raw[, "lopex_01.001"]

refr <- rrtm:::refractive_p45
talf <- rrtm:::p45_talf
t12 <- rrtm:::p45_t12
t21 <- rrtm:::p45_t21
kmat <- rrtm:::dataspec_p4

data_list <- list(
  nwl = length(obs),
  obs = obs,
  talf = talf,
  t12 = t12,
  t21 = t21,
  kmat = kmat
)

mod <- cmdstan_model(stanfile)
fit <- mod$sample(data_list)

draws <- as_draws_df(fit)
summarize_draws(draws)
