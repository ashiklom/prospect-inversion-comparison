/* Vectorized exponential integral E1 via GSL, for nimbleExternalCall. */
#ifdef __cplusplus
extern "C" {
#endif
void e1_gsl_vec(double* x, int n, double* out);
#ifdef __cplusplus
}
#endif
