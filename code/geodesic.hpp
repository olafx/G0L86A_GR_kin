#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace geodesic
{

namespace detail
{

template <typename Fields>
[[nodiscard]] double u_0
( const Fields& fields,
  double eps,
  const Vec3& u
)
{ double u_0_ = eps;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      u_0_ += u[i]*u[j]*fields.gamma_con[i][j];
  return sqrt(u_0_)/fields.alpha;
}

template <typename Fields>
[[nodiscard]] Vec3 rhs_x
( const Fields& fields,
  double eps,
  const Vec3& u
)
{ const double u_0_ = u_0(fields, eps, u);
  Vec3 dx_dt;
  for (size_t i = 0; i < 3; i++)
  { double a = 0;
    for (size_t j = 0; j < 3; j++)
      a += fields.gamma_con[i][j]*u[j];
    dx_dt[i] = a/u_0_-fields.beta_con[i];
  }
  return dx_dt;
}

template <typename Fields, typename Derivatives>
[[nodiscard]] Vec3 rhs_u
( const Fields& fields,
  const Derivatives& derivatives,
  double eps,
  const Vec3& u
)
{ const double u_0_ = u_0(fields, eps, u);
  Vec3 du_dt;
  for (size_t i = 0; i < 3; i++)
  { const double a = -fields.alpha*u_0_*derivatives.d_alpha[i];
    double b = 0, c = 0;
    for (size_t j = 0; j < 3; j++)
      b += u[j]*derivatives.d_beta_con[i][j];
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        c += u[j]*u[k]*derivatives.d_gamma_con[i][j][k];
    c /= -2*u_0_;
    du_dt[i] = a+b+c;
  };
  return du_dt;
}

} // namespace detail

// Describe the RHS of the geodesic equation, i.e. y'=rhs(y). We represent this
// y as a Mat23 instead of Vec6 for clarity, separating the position and
// velocity.
template <typename Metric>
[[nodiscard]] Mat23 rhs
( const Metric& metric,
  double eps,
  const Mat23& state,
  const finite_difference::Policy<3> auto& policy_fd
)
{ const Vec3& x = state.X;
  const Vec3& u = state.V;
  const auto f = metric.fields(x);
// TODO: This is numerical for now since analytical has not been implemented.
//   rhs will no longer need a fd_step_policy, and numerical params won't need
//   to be passed anymore.
  const auto d = metric.derivatives_numerical(policy_fd, x);
  return
  { detail::rhs_x(f, eps, u),
    detail::rhs_u(f, d, eps, u)
  };
}

template <typename Metric>
[[nodiscard]] Vec3 rhs_x
( const Metric& metric,
  double eps,
  const Vec3& x,
  const Vec3& u
)
{ const auto f = metric.fields(x);
  return detail::rhs_x(f, eps, u);
}

template <typename Metric>
[[nodiscard]] Vec3 rhs_u
( const Metric& metric,
  double eps,
  const Vec3& x,
  const Vec3& u,
  const finite_difference::Policy<3> auto& policy_fd
)
{ const auto f = metric.fields(x);
  const auto d = metric.derivatives_numerical(policy_fd, x);
  return detail::rhs_u(f, d, eps, u);
}

} // namespace geodesic
