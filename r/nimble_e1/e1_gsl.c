#include <gsl/gsl_sf_expint.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include "e1_gsl.h"

void e1_gsl_vec(double* x, int n, double* out) {
  gsl_sf_result res;
  for (int i = 0; i < n; i++) {
    /* The _e variant reports underflow (x >~ 700) instead of aborting. */
    int status = gsl_sf_expint_E1_e(x[i], &res);
    out[i] = (status == GSL_SUCCESS) ? res.val : (status == GSL_EUNDRFLW ? 0.0 : NAN);
  }
}
