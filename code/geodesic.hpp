#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"
#include "metric.hpp"
#include "solve.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace geodesic
{

namespace detail
{

// Defining the RHS of the geodesic equation.
// Sometimes we need dx/dt and du/dt simultaneously, sometimes separately.

template <typename MetricADM>
[[nodiscard]] double u_0
( const MetricADM& m,
  double eps,
  const Vec3& u
)
{ double u_0_ = eps;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      u_0_ += u[i]*u[j]*m.gamma_con[i][j];
  return sqrt(u_0_)/m.alpha;
}

template <typename MetricADM>
[[nodiscard]] Vec3 rhs_x
( const MetricADM& m,
  double eps,
  const Vec3& u
)
{ const double u_0_ = u_0(m, eps, u);
  Vec3 dx_dt;
  for (size_t i = 0; i < 3; i++)
  { double a = 0;
    for (size_t j = 0; j < 3; j++)
      a += m.gamma_con[i][j]*u[j];
    dx_dt[i] = a/u_0_-m.beta_con[i];
  }
  return dx_dt;
}

template <typename MetricADM, typename MetricADMDerivatives>
[[nodiscard]] Vec3 rhs_u
( const MetricADM& m,
  const MetricADMDerivatives& d,
  double eps,
  const Vec3& u
)
{ const double u_0_ = u_0(m, eps, u);
  Vec3 du_dt;
  for (size_t i = 0; i < 3; i++)
  { const double a = -m.alpha*u_0_*d.d_alpha[i];
    double b = 0, c = 0;
    for (size_t j = 0; j < 3; j++)
      b += u[j]*d.d_beta_con[i][j];
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        c += u[j]*u[k]*d.d_gamma_con[i][j][k];
    c /= -2*u_0_;
    du_dt[i] = a+b+c;
  };
  return du_dt;
}

} // namespace detail

// Define the geodesic problem as a type. The point of this abstraction is to
// absorb the problem parameters into the type, as to keep the same general
// high-level interface.

struct Problem
{
  const metric::Kerr::BoyerLindquist& metric;
  const finite_difference::policies::Simple& policy_fd;
  double eps;

  [[nodiscard]] Mat23 rhs
  ( const Vec3& x,
    const Vec3& u
  ) const
  { const auto m = metric.metric_ADM(x);
// TODO: This is numerical for now since analytical has not been implemented.
//   rhs will no longer need a fd_step_policy, and numerical params won't need
//   to be passed anymore. Need to design this all some way that both numerical
//   and analytical derivatives can work without too much code reuse.
    const auto d = metric.metric_ADM_derivatives_numerical(policy_fd, x);
    return
    { detail::rhs_x(m, eps, u),
      detail::rhs_u(m, d, eps, u)
    };
  }

  [[nodiscard]] Mat23 rhs
  ( const Mat23& state
  ) const
  { return rhs(state.X, state.V);
  }

  [[nodiscard]] Vec3 rhs_x
  ( const Vec3& x,
    const Vec3& u
  ) const
  { const auto m = metric.metric_ADM(x);
    return detail::rhs_x(m, eps, u);
  }

  [[nodiscard]] Vec3 rhs_u
  ( const Vec3& x,
    const Vec3& u
  ) const
  { const auto m = metric.metric_ADM(x);
    const auto d = metric.metric_ADM_derivatives_numerical(policy_fd, x);
    return detail::rhs_u(m, d, eps, u);
  }
};

////////////////////////////////////////////////////////////////////////////////

namespace schemes
{

// Define an integration scheme for the problem, using a `Stepper`. This deals
// with initialization, the way in which the `Stepper` is used for the problem,
// and how the internal integration state is converted to an output state. This
// abstraction also helps maintain a clean high-level generic interface.

template <typename Stepper>
struct Full
{
  Stepper stepper;

  void initialize
  ( const Problem&,
    Mat23&
  ) const
  {}

  void step
  ( const Problem& problem,
    Mat23& state
  ) const
  { auto rhs = [&](const Vec6& y)
    { return Vec6 {problem.rhs(Mat23 {y})}; };
    static_assert(solve::Fn_Stepper<Stepper, decltype(rhs), 6>);
    state = Mat23 {stepper(rhs, Vec6 {state})};
  }

  [[nodiscard]] Mat23 output
  ( const Problem&,
    const Mat23&,
    const Mat23& state
  ) const
  { return state;
  }
};

template <typename Stepper>
struct Split
{
  Stepper stepper;

// NOTE: Initial condition gives x^0 and u^0, compute u^(1/2) so leapfrogging
//   can continue.
  void initialize
  ( const Problem& problem,
    Mat23& state
  ) const
  { auto rhs_u = [&](const Vec3& u)
    { return problem.rhs_u(state.X, u); };
    static_assert(solve::Fn_Stepper<Stepper, decltype(rhs_u), 3>);
    auto half_stepper = stepper;
    half_stepper.dt *= .5;
    state.V = half_stepper(rhs_u, state.V);
  }

// NOTE: First u^(n+1/2) from u^(n-1/2) and x^(n), then x^(n+1) from x^(n) and
//   u^(n+1/2).
  void step
  ( const Problem& problem,
    Mat23& state
  ) const
  { 
    auto rhs_x = [&](const Vec3& x)
    { return problem.rhs_x(x, state.V); };
    auto rhs_u = [&](const Vec3& u)
    { return problem.rhs_u(state.X, u); };
    static_assert(solve::Fn_Stepper<Stepper, decltype(rhs_x), 3>);
    static_assert(solve::Fn_Stepper<Stepper, decltype(rhs_u), 3>);
    state.V = stepper(rhs_u, state.V);
    state.X = stepper(rhs_x, state.X);
  }

// NOTE: For output, already have x^(n), but get u^(n) estimate from u^(n+1/2)
//   and u^(n-1/2).
  [[nodiscard]] Mat23 output
  ( const Problem&,
    const Mat23& state_prev,
    const Mat23& state
  ) const
  { return {state.X, .5*(state_prev.V+state.V)};
  }
};

} // namespace schemes

} // namespace geodesic
