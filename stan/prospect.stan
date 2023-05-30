functions {
  real expint(real x) {
    real A;
    real B;
    A = log((0.56146 / x + 0.65) * (1 + x));
    B = x^4 * exp(7.7 * x) * (2 + x)^3.7;
    return (A^-7.7 + B)^-0.13;
  }

  real transfun(real k) {
    real trans;
    if (k <= 0) {
      trans = 1.0;
    } else {
      trans = (1 - k) * exp(-k) + k^2 * expint(k);
    }
    if (trans < 0) {
      trans = 0.0;
    }
    return trans;
  }

  real gpm(real N, real k, real talf, real t12, real t21) {
    real trans = transfun(k);
    real ralf = 1 - talf;
    real r12 = 1 - t12;
    real r21 = 1 - t21;
    real denom = 1 - r21 ^ 2 * trans ^ 2;
    real Ta = talf * trans * t21 / denom;
    real Ra = ralf + r21 * trans * Ta;

    real t = t12 * trans * t21 / denom;
    real r = r12 + r21 * trans * t;

    real Tsub;
    real Rsub;

    if (r + t >= 1) {
      Tsub = t / (t + (1 - t) * (N - 1));
      Rsub = 1 - Tsub;
    } else {
      real D = sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t));
      real r2 = r ^ 2;
      real t2 = t ^ 2;
      real va = (1 + r2 - t2 + D) / (2 * r);
      real vb = (1 - r2 + t2 + D) / (2 * t);
      real vbNN = vb ^ (N - 1);
      real vbNN2 = vbNN ^ 2;
      real va2 = va ^ 2;
      real denomx = va2 * vbNN2 - 1;
      Rsub = va * (vbNN2 - 1) / denomx;
      Tsub = vbNN * (va2 - 1) / denomx;
    }
    real denomy = 1 - Rsub * r;
    // Reflectance
    real RN = Ra + Ta * Rsub * t / denomy;
    // Transmittance
    // TN <- Ta * Tsub / denomy
    // result[1:2101,1] <- RN
    // result[1:2101,2] <- TN
    // returnType(double(2101, 2))
    return RN;
  }

  vector prospect4(real N, real Cab, real Cw, real Cm,
                   vector talf, vector t12, vector t21,
                   matrix kmat) {
    matrix[3,1] cc;
    cc[1,1] = Cab / N;
    cc[2,1] = Cw / N;
    cc[3,1] = Cm / N;
    vector[2101] k = to_vector(kmat * cc);
    vector[2101] result;
    for (i in 1:2101) {
        result[i] = gpm(N, k[i], talf[i], t12[i], t21[i]);
    }
    return result;
  }
}

data {
  int<lower=0> nwl;
  vector[nwl] obs;
  vector[nwl] talf;
  vector[nwl] t12;
  vector[nwl] t21;
  matrix[nwl, 3] kmat;
}

parameters {
  real<lower=1> N;
  real<lower=0> Cab;
  real<lower=0> Cw;
  real<lower=0> Cm;
  real<lower=0> rsd;
}

model {
    N ~ normal(1.4, 0.5);
    Cab ~ normal(40, 25);
    Cw ~ normal(0.01, 0.005);
    Cm ~ normal(0.01, 0.005);
    obs ~ normal(prospect4(N, Cab, Cw, Cm,
                           talf, t12, t21, kmat), rsd);
}
