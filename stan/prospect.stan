functions {
  real expint(real x) {
    real a;
    real b;
    a = log((0.56146 / x + 0.65) * (1 + x));
    b = x^4 * exp(7.7 * x) * (2 + x)^3.7;
    return (a^-7.7 + b)^-0.13;
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

  real gpm(real n, real k, data real talf, data real t12, data real t21) {
    real trans = transfun(k);
    real ralf = 1 - talf;
    real r12 = 1 - t12;
    real r21 = 1 - t21;
    real denom = 1 - r21 ^ 2 * trans ^ 2;
    real ta = talf * trans * t21 / denom;
    real ra = ralf + r21 * trans * ta;

    real t = t12 * trans * t21 / denom;
    real r = r12 + r21 * trans * t;

    real tsub;
    real rsub;

    if (r + t >= 1) {
      tsub = t / (t + (1 - t) * (n - 1));
      rsub = 1 - tsub;
    } else {
      real d = sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t));
      real r2 = r ^ 2;
      real t2 = t ^ 2;
      real va = (1 + r2 - t2 + d) / (2 * r);
      real vb = (1 - r2 + t2 + d) / (2 * t);
      real vbnn = vb ^ (n - 1);
      real vbnn2 = vbnn ^ 2;
      real va2 = va ^ 2;
      real denomx = va2 * vbnn2 - 1;
      rsub = va * (vbnn2 - 1) / denomx;
      tsub = vbnn * (va2 - 1) / denomx;
    }
    real denomy = 1 - rsub * r;
    // reflectance
    real rn = ra + ta * rsub * t / denomy;
    // transmittance
    // tn <- ta * tsub / denomy
    return rn;
  }

  vector prospect4(real n, real cab, real cw, real cm,
                   data vector talf, data vector t12, data vector t21,
                   data matrix kmat) {
    matrix[3,1] cc;
    int nwl = size(talf);
    cc[1,1] = cab / n;
    cc[2,1] = cw / n;
    cc[3,1] = cm / n;
    vector[nwl] k = to_vector(kmat * cc);
    vector[nwl] result;
    for (i in 1:nwl) {
        result[i] = gpm(n, k[i], talf[i], t12[i], t21[i]);
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
  real<lower=1> n;
  real<lower=0> cab;
  real<lower=0> cw;
  real<lower=0> cm;
  real<lower=0> rsd;
}

model {
    n ~ normal(1.4, 0.5);
    cab ~ normal(40, 25);
    cw ~ normal(0.01, 0.005);
    cm ~ normal(0.01, 0.005);
    vector[nwl] mod = prospect4(n, cab, cw, cm, talf, t12, t21, kmat);
    obs ~ normal(mod, rsd);
}
