  // PROSPECT family in Stan. #include'd inside a `functions {}` block.
  //
  // e1mode: 1 = exact E1 (external C++, boost::math::expint with an analytic
  //             gradient dE1/dx = -exp(-x)/x; see stan/expint.hpp)
  //         2 = Swamee & Ohija (2003) closed-form approximation

  vector e1_exact(vector x);

  vector e1_approx(vector x) {
    vector[rows(x)] a = log((0.56146 ./ x + 0.65) .* (1 + x));
    vector[rows(x)] b = pow(x, 4) .* exp(7.7 * x) .* pow(2 + x, 3.7);
    return pow(pow(a, -7.7) + b, -0.13);
  }

  // Generalized plate model. Returns an n x 1 matrix (reflectance) or, if
  // want_t == 1, an n x 2 matrix (reflectance, transmittance).
  // Assumes k > 0 everywhere (true whenever Cw > 0 or Cm > 0), so the
  // no-absorption branch of the reference implementation is not needed.
  matrix gpm(vector k, real N, vector talf, vector t12, vector t21,
             int e1mode, int want_t) {
    int n = rows(k);
    vector[n] e1 = e1mode == 1 ? e1_exact(k) : e1_approx(k);
    vector[n] tau = fmax((1 - k) .* exp(-k) + square(k) .* e1, 0);
    vector[n] ralf = 1 - talf;
    vector[n] r12 = 1 - t12;
    vector[n] r21 = 1 - t21;
    vector[n] denom = 1 - square(r21) .* square(tau);
    vector[n] Ta = talf .* tau .* t21 ./ denom;
    vector[n] Ra = ralf + r21 .* tau .* Ta;
    vector[n] t = t12 .* tau .* t21 ./ denom;
    vector[n] r = r12 + r21 .* tau .* t;
    vector[n] D = sqrt((1 + r + t) .* (1 + r - t) .* (1 - r + t) .* (1 - r - t));
    vector[n] va = (1 + square(r) - square(t) + D) ./ (2 * r);
    vector[n] vb = (1 - square(r) + square(t) + D) ./ (2 * t);
    vector[n] vbNN = pow(vb, N - 1);
    vector[n] vbNN2 = square(vbNN);
    vector[n] denomx = square(va) .* vbNN2 - 1;
    vector[n] Rsub = va .* (vbNN2 - 1) ./ denomx;
    vector[n] denomy = 1 - Rsub .* r;
    matrix[n, want_t == 1 ? 2 : 1] out;
    out[, 1] = Ra + Ta .* Rsub .* t ./ denomy;
    if (want_t == 1) {
      vector[n] Tsub = vbNN .* (square(va) - 1) ./ denomx;
      out[, 2] = Ta .* Tsub ./ denomy;
    }
    return out;
  }

  // PROSPECT reflectance for constituent vector C (columns of K)
  vector prospect_refl(real N, vector C, matrix K, vector talf, vector t12,
                       vector t21, int e1mode) {
    return gpm(K * C / N, N, talf, t12, t21, e1mode, 0)[, 1];
  }

  matrix prospect_rt(real N, vector C, matrix K, vector talf, vector t12,
                     vector t21, int e1mode) {
    return gpm(K * C / N, N, talf, t12, t21, e1mode, 1);
  }
