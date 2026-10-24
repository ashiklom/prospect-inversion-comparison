// PROSPECT inversion with a heteroskedastic, AR(1)-autocorrelated residual:
//
//   obs_i = R_i(theta) + s_i * z_i,   s_i = sigma0 + sigma1 * R_i(theta)
//   z_1 ~ N(0, 1),  z_i | z_{i-1} ~ N(phi * z_{i-1}, sqrt(1 - phi^2))
//
// i.e. a stationary unit-variance AR(1) process across adjacent (1 nm)
// bands, scaled by a reflectance-dependent SD. The likelihood is exact and
// O(n): the AR(1) precision matrix is tridiagonal. Requires a contiguous
// wavelength grid.
functions {
#include prospect_functions.stan
}
data {
  int<lower=2> nwl;
  int<lower=1> nc;
  vector[nwl] obs;
  matrix[nwl, nc] K;
  vector[nwl] talf;
  vector[nwl] t12;
  vector[nwl] t21;
  vector[nc] prior_meanlog;
  vector<lower=0>[nc] prior_sdlog;
  real prior_N_mean;
  real<lower=0> prior_N_sd;
  real prior_N_lower;
  real<lower=0> prior_sigma_scale;
  int<lower=1, upper=2> e1mode;
}
parameters {
  real<lower=prior_N_lower> N;
  vector<lower=0>[nc] C;
  real<lower=0> sigma0;
  real<lower=0> sigma1;
  real<lower=0, upper=1> phi;
}
model {
  vector[nwl] R = prospect_refl(N, C, K, talf, t12, t21, e1mode);
  vector[nwl] s = sigma0 + sigma1 * R;
  vector[nwl] z = (obs - R) ./ s;

  N ~ normal(prior_N_mean, prior_N_sd);
  C ~ lognormal(prior_meanlog, prior_sdlog);
  sigma0 ~ normal(0, prior_sigma_scale);
  sigma1 ~ normal(0, 0.2);
  phi ~ beta(1, 1);

  z[1] ~ std_normal();
  z[2:nwl] ~ normal(phi * z[1:(nwl - 1)], sqrt(1 - square(phi)));
  target += -sum(log(s));  // Jacobian of z = (obs - R) / s
}
