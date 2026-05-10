#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"
#include "electromagnetic.hpp"

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

template <typename Fields>
[[nodiscard]] Vec3 lorentz_force
( const Fields& f,
  const em::Fields& em,
  const Vec3& u,
  double u_0_
)
{ const double sq = f.sqrt_gamma;
  double v[3] = {0.0, 0.0, 0.0};
  for (size_t j = 0; j < 3; j++)
    for (size_t l = 0; l < 3; l++)
      v[j] += f.gamma_con[j][l] * u[l];
  for (size_t j = 0; j < 3; j++) v[j] /= u_0_;

  const double cross_vel_B[3] = {
    sq*(v[1]*em.B[2] - v[2]*em.B[1]),
    sq*(v[2]*em.B[0] - v[0]*em.B[2]),
    sq*(v[0]*em.B[1] - v[1]*em.B[0])
  };

  Vec3 force;
  for (size_t i = 0; i < 3; i++)
  { double elec = 0;
    for (size_t j = 0; j < 3; j++)
      elec += f.alpha * f.gamma_cov[i][j] * em.D[j];
    force[i] = elec + cross_vel_B[i]/u_0_;
  }
  return force;
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

// EM-aware rhs_u: geodesic acceleration plus Lorentz force.
template <typename Metric, em::EMField EM>
[[nodiscard]] Vec3 rhs_u
( const Metric& metric,
  double eps,
  double q_over_m,
  const EM& em_field,
  const Vec3& x,
  const Vec3& u,
  const finite_difference::Policy<3> auto& policy_fd
)
{ const auto f = metric.fields(x);
  const auto d = metric.derivatives_numerical(policy_fd, x);
  Vec3 du_dt = detail::rhs_u(f, d, eps, u);

  if (q_over_m != 0.0)
  { const em::Fields em = em_field(x);
    const double u_0_ = detail::u_0(f, eps, u);
    const Vec3 force = detail::lorentz_force(f, em, u, u_0_);
    for (size_t i = 0; i < 3; i++)
      du_dt[i] += q_over_m * force[i];
  }

  return du_dt;
}

// EM-aware overload: adds Lorentz force for charged particles (q_over_m != 0).
// For q_over_m = 0 or em_field = em::Vacuum{}, reduces to pure geodesic motion.
template <typename Metric, em::EMField EM>
[[nodiscard]] Mat23 rhs
( const Metric& metric,
  double eps,
  double q_over_m,
  const EM& em_field,
  const Mat23& state,
  const finite_difference::Policy<3> auto& policy_fd
)
{ const Vec3& x = state.X;
  const Vec3& u = state.V;
  const auto f = metric.fields(x);
  const auto d = metric.derivatives_numerical(policy_fd, x);

  Mat23 result
  { detail::rhs_x(f, eps, u),
    detail::rhs_u(f, d, eps, u)
  };

  if (q_over_m != 0.0)
  { const em::Fields em = em_field(x);
    const double u_0_ = detail::u_0(f, eps, u);
    const Vec3 force = detail::lorentz_force(f, em, u, u_0_);
    for (size_t i = 0; i < 3; i++)
      result.V[i] += q_over_m * force[i];
  }

  return result;
}

} // namespace geodesic
