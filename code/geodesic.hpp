#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"
#include "solve.hpp"
#include "electromagnetic.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace geodesic
{

// Geodesics solve particle trajectories. Describe all the particle properties
// that may vary within a given spacetime with a given electromagnetic field.

namespace particle
{

struct Neutral
{ double eps;
};

struct Charged
{ double eps;
  double q_over_m;
};

template <typename Particle>
concept Kind = requires(const Particle& p)
{ { p.eps } -> std::convertible_to<double>;
};

template <typename Particle>
concept ChargedKind =
  Kind<Particle> &&
  requires(const Particle& p)
  { { p.q_over_m } -> std::convertible_to<double>;
  };

} // namespace particle

////////////////////////////////////////////////////////////////////////////////

namespace detail
{

// Defining the RHS of the geodesic equation.
//
// NOTE: Sometimes we need dx/dt and du/dt simultaneously, sometimes separately.

[[nodiscard]] double u_0_con
( const metric::MetricADMGeo& m,
  double eps,
  const Vec3& u
)
{ double u_0_con_ = eps;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      u_0_con_ += u[i]*u[j]*m.gamma_con[i][j];
  return sqrt(u_0_con_)/m.alpha;
}

[[nodiscard]] Vec3 rhs_x
( const metric::MetricADMGeo& m,
  double eps,
  double u_0_con,
  const Vec3& u
)
{ Vec3 dx_dt;
  for (size_t i = 0; i < 3; i++)
  { double a = 0;
    for (size_t j = 0; j < 3; j++)
      a += m.gamma_con[i][j]*u[j];
    dx_dt[i] = a/u_0_con-m.beta_con[i];
  }
  return dx_dt;
}

// This RHS of du/dt lacks the Lorentz term, that is separate.
[[nodiscard]] Vec3 rhs_u
( const metric::MetricADMGeo& m,
  const metric::MetricADMDerivativesGeo& d,
  double eps,
  double u_0_con,
  const Vec3& u
)
{ Vec3 du_dt;
  for (size_t i = 0; i < 3; i++)
  { const double a = -m.alpha*u_0_con*d.d_alpha[i];
    double b = 0, c = 0;
    for (size_t j = 0; j < 3; j++)
      b += u[j]*d.d_beta_con[i][j];
    for (size_t j = 0; j < 3; j++)
      for (size_t k = 0; k < 3; k++)
        c += u[j]*u[k]*d.d_gamma_con[i][j][k];
    c /= -2*u_0_con;
    du_dt[i] = a+b+c;
  };
  return du_dt;
}

// This Lorentz term of the RHS of du/dt lacks q/m, that is separate.
[[nodiscard]] Vec3 rhs_Lorentz
( const metric::MetricADMGeo& m_geo,
  const metric::MetricADMLorentz& m_Lorentz,
  const em::Fields& em_f,
  double u_0_con,
  const Vec3& u
)
{ Vec3 v {0, 0, 0};
  for (size_t j = 0; j < 3; j++)
    for (size_t l = 0; l < 3; l++)
      v[j] += m_geo.gamma_con[j][l]*u[l];
  // for (size_t j = 0; j < 3; j++)
  //   v[j] /= u_0_con_;
  Vec3 cross_v_B = cross(v, em_f.B);

  Vec3 force;
  for (size_t i = 0; i < 3; i++)
  { double elec = 0;
    for (size_t j = 0; j < 3; j++)
      elec += m_geo.alpha*m_Lorentz.gamma_cov[i][j]*em_f.D[j];
    // force[i] = elec+m.sqrt_gamma/u_0_con_*cross_vel_B[i];
    force[i] = elec-m_Lorentz.sqrt_gamma/u_0_con*cross_v_B[i];
  }
  return force;
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////

// Define the geodesic problem (including an electromagnetic field) as a type.
//
// The point of this abstraction is to absorb the problem parameters into the
// type, as to keep the same general high-level interface.
//
// TODO: Derivatives are numerical for now since analytical has not been
//   implemented. Need to design this all some way that both numerical and
//   analytical derivatives can work without too much code reuse.

template <typename Metric, particle::Kind Particle, typename EMField>
struct Problem
{
  const Metric& metric;
  const EMField& em_field;
  const finite_difference::policies::Simple& policy_fd;
  Particle p;

  [[nodiscard]] Mat23 rhs
  ( const Vec3& x,
    const Vec3& u
  ) const
  {
    if constexpr (particle::ChargedKind<Particle>)
    { const auto m = metric.template eval<true>(x);
      const auto d_m = metric.eval_derivatives_numerical(policy_fd, x);
      const double u_0_con_ = detail::u_0_con(m.geo, p.eps, u);
      Vec3 du_dt = detail::rhs_u(m.geo, d_m, p.eps, u_0_con_, u);
      if (p.q_over_m)
        du_dt += p.q_over_m*detail::rhs_Lorentz(
          m.geo, m.lorentz, em_field(m, x), u_0_con_, u);
      const Vec3 dx_dt = detail::rhs_x(m.geo, p.eps, u_0_con_, u);
      return {dx_dt, du_dt};
    }
    else
    { const auto m = metric.template eval<false>(x);
      const auto d_m = metric.eval_derivatives_numerical(policy_fd, x);
      const double u_0_con_ = detail::u_0_con(m, p.eps, u);
      Vec3 du_dt = detail::rhs_u(m, d_m, p.eps, u_0_con_, u);
      const Vec3 dx_dt = detail::rhs_x(m, p.eps, u_0_con_, u);
      return {dx_dt, du_dt};
    }
  }

  [[nodiscard]] Vec3 rhs_x
  ( const Vec3& x,
    const Vec3& u
  ) const
  { const auto m = metric.template eval<false>(x);
    const double u_0_con_ = detail::u_0_con(m, p.eps, u);
    return detail::rhs_x(m, p.eps, u_0_con_, u);
  }

  [[nodiscard]] Vec3 rhs_u
  ( const Vec3& x,
    const Vec3& u
  ) const
  {
    if constexpr (particle::ChargedKind<Particle>)
    { const auto m = metric.template eval<true>(x);
      const auto d = metric.eval_derivatives_numerical(policy_fd, x);
      const double u_0_con_ = detail::u_0_con(m.geo, p.eps, u);
      Vec3 du_dt = detail::rhs_u(m.geo, d, p.eps, u_0_con_, u);
      if (p.q_over_m)
        du_dt += p.q_over_m*detail::rhs_Lorentz(
          m.geo, m.lorentz, em_field(m, x), u_0_con_, u);
      return du_dt;
    }
    else
    { const auto m = metric.template eval<false>(x);
      const auto d = metric.eval_derivatives_numerical(policy_fd, x);
      const double u_0_con_ = detail::u_0_con(m, p.eps, u);
      return detail::rhs_u(m, d, p.eps, u_0_con_, u);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

namespace schemes
{

// Define an integration scheme for the `Problem`, using a `Stepper`. This deals
// with initialization, the way in which the `Stepper` is used for the problem,
// and how the internal integration state is converted to an output state. This
// abstraction helps maintain a clean high-level generic interface.

template <typename Stepper>
struct Full
{
  Stepper stepper;

  template <typename Metric, typename Particle, typename Field>
  void initialize
  ( const Problem<Metric, Particle, Field>&,
    Mat23& state
  ) const
  {}

  template <typename Metric, typename Particle, typename Field>
  void step
  ( const Problem<Metric, Particle, Field>& problem,
    Mat23& state
  ) const
  { auto rhs = [&](const Vec6& y)
    { const Mat23& y_ = y;
      return Vec6 {problem.rhs(y_.X, y_.V)};
    };
    static_assert(solve::Fn_Stepper<Stepper, decltype(rhs), 6>);
    state = stepper(rhs, Vec6 {state});
  }

  template <typename Metric, typename Particle, typename Field>
  [[nodiscard]] Mat23 output
  ( const Problem<Metric, Particle, Field>&,
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

// NOTE: Initial condition gives x^0 and u^0. Compute u^(1/2) so leapfrogging
//   can continue.
  template <typename Metric, typename Particle, typename Field>
  void initialize
  ( const Problem<Metric, Particle, Field>& problem,
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
  template <typename Metric, typename Particle, typename Field>
  void step
  ( const Problem<Metric, Particle, Field>& problem,
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
  template <typename Metric, typename Particle, typename Field>
  [[nodiscard]] Mat23 output
  ( const Problem<Metric, Particle, Field>&,
    const Mat23& state_prev,
    const Mat23& state
  ) const
  { return {state.X, .5*(state_prev.V+state.V)};
  }
};

} // namespace schemes

} // namespace geodesic
