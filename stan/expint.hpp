// Exact exponential integral E1(x) for Stan, with an analytic gradient.
//
// Value: boost::math::expint(1, x) (Boost ships with Stan Math).
// Gradient: dE1/dx = -exp(-x) / x, attached with make_callback_var so the
// special function itself never needs to be differentiated.
//
// Stan calls `e1_exact(x, pstream__)` unqualified from inside the model
// namespace, so a global-namespace template is found for every model.
#ifndef PROSPECT_EXPINT_HPP
#define PROSPECT_EXPINT_HPP

#include <stan/math.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <cmath>
#include <ostream>

namespace prospect_detail {
// Never throw: return inf / 0 / NaN so that wild proposals (e.g. during
// step-size initialization) are rejected rather than aborting the chain.
// Stan treats only std::domain_error as recoverable.
using e1_policy = boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::pole_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>>;
inline double e1(double x) { return boost::math::expint(1, x, e1_policy()); }
}  // namespace prospect_detail

template <typename T, stan::require_eigen_col_vector_t<T>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T>, -1, 1> e1_exact(
    const T& x, std::ostream* pstream__) {
  const auto& xr = stan::math::to_ref(x);
  const Eigen::Index n = xr.size();
  if constexpr (stan::is_var<stan::value_type_t<T>>::value) {
    Eigen::Matrix<stan::math::var, -1, 1> out(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const stan::math::var xi = xr.coeff(i);
      const double v = xi.val();
      const double grad = -std::exp(-v) / v;
      out.coeffRef(i) = stan::math::make_callback_var(
          prospect_detail::e1(v),
          [xi, grad](auto& vi) mutable { xi.adj() += vi.adj() * grad; });
    }
    return out;
  } else {
    Eigen::Matrix<double, -1, 1> out(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      out.coeffRef(i) = prospect_detail::e1(static_cast<double>(xr.coeff(i)));
    }
    return out;
  }
}

#endif
