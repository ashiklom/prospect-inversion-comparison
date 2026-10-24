# Vectorized base-R implementation of the PROSPECT family (used by the
# BayesianTools backend and as an R-side reference).
#
# A PROSPECT version is defined by a fit-input table (see
# data/inputs/<version>/<sample>.csv) holding talf, t12, t21 and one
# specific-absorption column per constituent.

e1_exact <- function(x) expint::expint_E1(x)

# Swamee & Ohija (2003) approximation to E1(x)
e1_approx <- function(x) {
  a <- log((0.56146 / x + 0.65) * (1 + x))
  b <- x^4 * exp(7.7 * x) * (2 + x)^3.7
  (a^-7.7 + b)^-0.13
}

transmissivity <- function(k, e1 = c("exact", "approx")) {
  e1 <- match.arg(e1)
  e1fun <- switch(e1, exact = e1_exact, approx = e1_approx)
  tau <- rep(1, length(k))
  pos <- k > 0
  kp <- k[pos]
  tau[pos] <- (1 - kp) * exp(-kp) + kp^2 * e1fun(kp)
  pmax(tau, 0)
}

# Generalized plate model (Allen et al. 1969; Stokes 1862)
gpm <- function(k, N, talf, t12, t21, e1 = "exact") {
  tau <- transmissivity(k, e1)

  ralf <- 1 - talf
  r12 <- 1 - t12
  r21 <- 1 - t21
  denom <- 1 - r21^2 * tau^2
  Ta <- talf * tau * t21 / denom
  Ra <- ralf + r21 * tau * Ta

  t <- t12 * tau * t21 / denom
  r <- r12 + r21 * tau * t

  D <- sqrt(pmax((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), 0))
  va <- (1 + r^2 - t^2 + D) / (2 * r)
  vb <- (1 - r^2 + t^2 + D) / (2 * t)
  vbNN <- vb^(N - 1)
  denomx <- va^2 * vbNN^2 - 1
  Rsub <- va * (vbNN^2 - 1) / denomx
  Tsub <- vbNN * (va^2 - 1) / denomx

  noabs <- (r + t) >= 1
  if (any(noabs)) {
    tn <- t[noabs]
    Tsub[noabs] <- tn / (tn + (1 - tn) * (N - 1))
    Rsub[noabs] <- 1 - Tsub[noabs]
  }
  opaque <- (t <= 0 | r <= 0) & !noabs
  Rsub[opaque] <- 0
  Tsub[opaque] <- 0

  denomy <- 1 - Rsub * r
  list(
    R = Ra + Ta * Rsub * t / denomy,
    T = Ta * Tsub / denomy
  )
}

#' PROSPECT forward model
#'
#' @param N structure parameter
#' @param C constituent vector, ordered as the columns of `K`
#' @param inputs list with `K` (n_wl x n_constituents), `talf`, `t12`, `t21`
prospect <- function(N, C, inputs, e1 = "exact") {
  k <- drop(inputs$K %*% C) / N
  gpm(k, N, inputs$talf, inputs$t12, inputs$t21, e1)
}
