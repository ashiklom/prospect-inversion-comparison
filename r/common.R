# Shared helpers for the R-based backends (BayesianTools, Nimble, Stan).
# Assumes the working directory is the repository root.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

read_config <- function(name) {
  fromJSON(file.path("config", paste0(name, ".json")), simplifyVector = TRUE)
}

#' Load the fit inputs for one sample
#'
#' Returns a list with wavelength, obs (reflectance), obs_T (held-out
#' transmittance), K, talf, t12, t21, constituents, and prior vectors.
load_inputs <- function(sample_id, version = "prospectd") {
  models <- read_config("models")
  constituents <- models[[version]]$constituents
  d <- fread(file.path("data", "inputs", version, paste0(sample_id, ".csv")))
  priors <- read_config("priors")
  cp <- priors$constituents[constituents]
  list(
    sample_id = sample_id,
    version = version,
    constituents = constituents,
    wavelength = d$wavelength,
    obs = d$R,
    obs_T = d$T,
    K = as.matrix(d[, ..constituents]),
    talf = d$talf,
    t12 = d$t12,
    t21 = d$t21,
    prior_meanlog = vapply(cp, `[[`, numeric(1), "meanlog"),
    prior_sdlog = vapply(cp, `[[`, numeric(1), "sdlog"),
    prior_N = priors$N,
    prior_sigma_scale = priors$sigma$scale
  )
}

parse_args <- function(defaults = list(sample = "synthetic", errmodel = "iid",
                                        e1 = "exact", version = "prospectd")) {
  args <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    out[[key]] <- args[i + 1]
    i <- i + 2
  }
  out
}

fit_dir <- function(backend, args) {
  tag <- args$errmodel
  if (args$e1 != "exact") tag <- paste0(tag, "-", args$e1)
  d <- file.path("results", "fits", backend, tag, args$sample)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

#' Write standardized fit outputs
#'
#' @param draws data.frame with columns chain, draw, N, <constituents>, sigma, ...
write_fit <- function(dir, draws, meta) {
  fwrite(draws, file.path(dir, "draws.csv.gz"))
  write_json(meta, file.path(dir, "meta.json"), auto_unbox = TRUE, pretty = TRUE,
             digits = NA)
  message("Wrote ", dir)
}

sampling_config <- function(backend) read_config("sampling")[[backend]]

#' Shared initial values: prior medians with log-scale jitter
make_inits <- function(inp, chains, seed) {
  ini <- read_config("sampling")$init
  lapply(seq_len(chains), function(ch) {
    set.seed(seed + ch)
    list(N = ini$N,
         C = exp(inp$prior_meanlog + runif(length(inp$prior_meanlog), -ini$jitter, ini$jitter)),
         sigma = ini$sigma)
  })
}

#' Common metadata fields
base_meta <- function(backend, args, inp, settings, t_setup, t_sampling, extra = list()) {
  c(list(
    backend = backend,
    sample_id = args$sample,
    errmodel = args$errmodel,
    e1 = args$e1,
    version = args$version,
    n_wavelengths = length(inp$obs),
    constituents = inp$constituents,
    settings = settings,
    time_setup_s = t_setup,
    time_sampling_s = t_sampling,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    platform = R.version.string
  ), extra)
}

elapsed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  force(expr)
  proc.time()[["elapsed"]] - t0
}
