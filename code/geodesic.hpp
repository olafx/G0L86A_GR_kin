#pragma once

#include "util.hpp"
#include "common.hpp"
#include "metric.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace geodesic
{

// Describe the RHS of the geodesic equation, i.e. y'=rhs(y). We represent this
// y as a Mat23 instead of Vec6 for clarity, separating the position and
// velocity.
template <typename Metric>
[[nodiscard]] Mat23 rhs
( const metric::Kerr::Params& params_phys,
  const Metric& metric,
  double eps,
  const Mat23& state,
  const finite_difference::Policy<3> auto& policy_fd
)
{ const Vec3& x = state.X;
  const Vec3& u = state.V;
  const auto f = metric.fields(params_phys, x);
// TODO: This is numerical for now since analytical has not been implemented.
//   rhs will no longer need a fd_step_policy, and numerical params won't need
//   to be passed anymore.
  const auto d = metric.derivatives_numerical(params_phys, policy_fd, x);

  double u0 = eps;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      u0 += u[i]*u[j]*f.gamma_con[i][j];
  u0 = sqrt(u0)/f.alpha;

  Vec3 dx_dt;
  for (size_t i = 0; i < 3; i++)
  { double a = 0;
    for (size_t j = 0; j < 3; j++)
      a += f.gamma_con[i][j]*u[j];
    dx_dt[i] = a/u0-f.beta_con[i];
  }

  Vec3 du_dt;
  for (size_t i = 0; i < 3; i++)
  { const double a = -f.alpha*u0*d.d_alpha[i];
    double b = 0;
    for (size_t j = 0; j < 3; j++)
      b += u[j]*d.d_beta_con[i][j];
    double c = 0;
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        c += u[j]*u[k]*d.d_gamma_con[i][j][k];
    c /= -2*u0;
    du_dt[i] = a+b+c;
  }

  return {dx_dt, du_dt};
}

} // namespace geodesic
