// PROSPECT inversion with a single wavelength-independent residual SD.
functions {
#include prospect_functions.stan
}
data {
  int<lower=1> nwl;
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
  real<lower=0> sigma;
}
model {
  // Truncation normalizing constants are fixed and omitted.
  N ~ normal(prior_N_mean, prior_N_sd);
  C ~ lognormal(prior_meanlog, prior_sdlog);
  sigma ~ normal(0, prior_sigma_scale);
  obs ~ normal(prospect_refl(N, C, K, talf, t12, t21, e1mode), sigma);
}
