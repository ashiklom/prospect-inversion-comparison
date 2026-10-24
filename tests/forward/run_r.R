# Evaluate the base-R, NIMBLE and Stan PROSPECT implementations on the
# shared test cases; write tests/forward/out/<impl>_<e1>.csv
# Run from the repository root.
suppressPackageStartupMessages({
  library(jsonlite); library(data.table); library(cmdstanr)
})
source("r/prospect.R")
source("r/prospect_nimble.R")

cases <- fromJSON("tests/forward/cases.json")
coef <- fread("data/coefficients/prospectd.csv")
K <- as.matrix(coef[, cases$constituents, with = FALSE])
inp <- list(K = K, talf = coef$talf, t12 = coef$t12, t21 = coef$t21)
cs <- cases$cases
outdir <- "tests/forward/out"

write_out <- function(res, impl, e1) {
  dt <- rbindlist(lapply(seq_along(res), function(i) {
    data.table(case = cs$name[i], wavelength = coef$wavelength,
               R = res[[i]][, 1], T = res[[i]][, 2])
  }))
  fwrite(dt, file.path(outdir, sprintf("%s_%s.csv", impl, e1)))
}

for (e1 in c("exact", "approx")) {
  # base R
  res <- lapply(seq_len(nrow(cs)), function(i) {
    o <- prospect(cs$N[i], cs$C[[i]], inp, e1 = e1); cbind(o$R, o$T)
  })
  write_out(res, "r", e1)
}

# NIMBLE (compiled)
cnim <- compileNimble(prospect_rt_nim)
for (e1 in c("exact", "approx")) {
  mode <- if (e1 == "exact") 1L else 2L
  res <- lapply(seq_len(nrow(cs)), function(i) {
    cnim(cs$N[i], cs$C[[i]], K, inp$talf, inp$t12, inp$t21, mode)
  })
  write_out(res, "nimble", e1)
}

# Stan
set_cmdstan_path(file.path(Sys.getenv("CONDA_PREFIX"), "bin", "cmdstan"))
dir.create("stan/build", showWarnings = FALSE)
mod <- cmdstan_model("stan/prospect_forward.stan", include_paths = "stan",
                     user_header = normalizePath("stan/expint.hpp"), dir = "stan/build")
for (e1 in c("exact", "approx")) {
  dat <- list(nwl = nrow(K), nc = ncol(K), ncase = nrow(cs), K = K,
              talf = inp$talf, t12 = inp$t12, t21 = inp$t21,
              N = as.array(cs$N), C = do.call(rbind, cs$C),
              e1mode = if (e1 == "exact") 1L else 2L)
  fit <- mod$sample(data = dat, fixed_param = TRUE, chains = 1, iter_sampling = 1,
                    iter_warmup = 0, refresh = 0, show_messages = FALSE,
                    sig_figs = 18)
  rt <- fit$draws("rt", format = "draws_matrix")
  res <- lapply(seq_len(nrow(cs)), function(i) {
    cbind(as.numeric(rt[1, sprintf("rt[%d,%d,1]", i, seq_len(nrow(K)))]),
          as.numeric(rt[1, sprintf("rt[%d,%d,2]", i, seq_len(nrow(K)))]))
  })
  write_out(res, "stan", e1)
}
message("R-side forward tests written")
