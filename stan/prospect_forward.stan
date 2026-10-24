// Forward-model evaluation for tests/forward (run with fixed_param)
functions {
#include prospect_functions.stan
}
data {
  int<lower=1> nwl;
  int<lower=1> nc;
  int<lower=1> ncase;
  matrix[nwl, nc] K;
  vector[nwl] talf;
  vector[nwl] t12;
  vector[nwl] t21;
  array[ncase] real N;
  array[ncase] vector[nc] C;
  int<lower=1, upper=2> e1mode;
}
generated quantities {
  array[ncase] matrix[nwl, 2] rt;
  for (i in 1:ncase) rt[i] = prospect_rt(N[i], C[i], K, talf, t12, t21, e1mode);
}
