# PROSPECT in the NIMBLE DSL.
#
# E1 is either exact (GSL `gsl_sf_expint_E1`, via nimbleExternalCall; NIMBLE's
# samplers used here are gradient-free so no derivative is required) or the
# Swamee-Ohija approximation written directly in the DSL.
#
# Requires r/nimble_e1/e1_gsl.o (built by r/nimble_e1/build.sh).

suppressPackageStartupMessages(library(nimble))

nimble_e1_dir <- normalizePath(file.path("r", "nimble_e1"))
if (!file.exists(file.path(nimble_e1_dir, "e1_gsl.o"))) {
  system2(file.path(nimble_e1_dir, "build.sh"))
}

e1_gsl_vec <- nimbleExternalCall(
  function(x = double(1), n = integer(), out = double(1)) {},
  Cfun = "e1_gsl_vec",
  headerFile = file.path(nimble_e1_dir, "e1_gsl.h"),
  oFile = file.path(nimble_e1_dir, "e1_gsl.o"),
  returnType = void()
)

# Core model: returns an nwl x 2 matrix (reflectance, transmittance).
# e1mode: 1 = exact (GSL), 2 = Swamee-Ohija approximation.
prospect_rt_nim <- nimbleFunction(
  run = function(N = double(0), C = double(1), K = double(2),
                 talf = double(1), t12 = double(1), t21 = double(1),
                 e1mode = integer(0)) {
    returnType(double(2))
    nwl <- length(talf)
    nc <- length(C)
    k <- numeric(nwl)
    for (i in 1:nwl) {
      ksum <- 0
      for (j in 1:nc) ksum <- ksum + K[i, j] * C[j]
      k[i] <- ksum / N
    }
    e1 <- numeric(nwl)
    if (e1mode == 1) {
      e1_gsl_vec(k, nwl, e1)
    } else {
      for (i in 1:nwl) {
        a <- log((0.56146 / k[i] + 0.65) * (1 + k[i]))
        b <- k[i]^4 * exp(7.7 * k[i]) * (2 + k[i])^3.7
        e1[i] <- (a^(-7.7) + b)^(-0.13)
      }
    }
    out <- matrix(0, nrow = nwl, ncol = 2)
    for (i in 1:nwl) {
      ki <- k[i]
      tau <- 1
      if (ki > 0) tau <- (1 - ki) * exp(-ki) + ki^2 * e1[i]
      if (tau < 0) tau <- 0
      ralf <- 1 - talf[i]
      r12 <- 1 - t12[i]
      r21 <- 1 - t21[i]
      denom <- 1 - r21^2 * tau^2
      Ta <- talf[i] * tau * t21[i] / denom
      Ra <- ralf + r21 * tau * Ta
      t <- t12[i] * tau * t21[i] / denom
      r <- r12 + r21 * tau * t
      if (r + t >= 1) {
        Tsub <- t / (t + (1 - t) * (N - 1))
        Rsub <- 1 - Tsub
      } else if (t <= 0 | r <= 0) {
        Tsub <- 0
        Rsub <- 0
      } else {
        D2 <- (1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t)
        if (D2 < 0) D2 <- 0
        D <- sqrt(D2)
        va <- (1 + r^2 - t^2 + D) / (2 * r)
        vb <- (1 - r^2 + t^2 + D) / (2 * t)
        vbNN <- vb^(N - 1)
        denomx <- va^2 * vbNN^2 - 1
        Rsub <- va * (vbNN^2 - 1) / denomx
        Tsub <- vbNN * (va^2 - 1) / denomx
      }
      denomy <- 1 - Rsub * r
      out[i, 1] <- Ra + Ta * Rsub * t / denomy
      out[i, 2] <- Ta * Tsub / denomy
    }
    return(out)
  }
)

prospect_refl_nim <- nimbleFunction(
  run = function(N = double(0), C = double(1), K = double(2),
                 talf = double(1), t12 = double(1), t21 = double(1),
                 e1mode = integer(0)) {
    returnType(double(1))
    rt <- prospect_rt_nim(N, C, K, talf, t12, t21, e1mode)
    return(rt[, 1])
  }
)

# Vectorized iid normal likelihood (avoids one graph node per wavelength)
dnorm_iid <- nimbleFunction(
  run = function(x = double(1), mean = double(1), sd = double(0),
                 log = integer(0, default = 0)) {
    returnType(double(0))
    n <- length(x)
    z <- (x - mean) / sd
    ll <- -0.5 * sum(z^2) - n * log(sd) - 0.5 * n * log(2 * pi)
    if (log) return(ll) else return(exp(ll))
  }
)

rnorm_iid <- nimbleFunction(
  run = function(n = integer(0), mean = double(1), sd = double(0)) {
    returnType(double(1))
    m <- length(mean)
    out <- numeric(m)
    for (i in 1:m) out[i] <- rnorm(1, mean[i], sd)
    return(out)
  }
)

registerDistributions(list(
  dnorm_iid = list(
    BUGSdist = "dnorm_iid(mean, sd)",
    types = c("value = double(1)", "mean = double(1)", "sd = double(0)")
  )
), verbose = FALSE)
