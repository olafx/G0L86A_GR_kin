#pragma once

#include <cstddef>
#include <concepts>
#include <type_traits>

#include "util.hpp"
#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace solve
{

// We only consider autonomous ODEs for now, so y'=f(y). `Fn_RHS` is such a
// function f.
template <typename Fn, size_t N>
concept Fn_RHS =
  std::invocable<const Fn&, const Vec<N>&> &&
  std::same_as<
    std::invoke_result_t<const Fn&, const Vec<N>&>,
    Vec<N>>;

// A `Stepper` is a function with some params, an invocable struct.
template <typename Stepper, typename Fn, size_t N>
concept Fn_Stepper =
  Fn_RHS<Fn, N> &&
  std::invocable<const Stepper&, const Fn&, const Vec<N>&> &&
  std::same_as<
    std::invoke_result_t<const Stepper&, const Fn&, const Vec<N>&>,
    Vec<N>>;

// `Stepper`: 3/8 rule 4th order Runge-Kutta
struct RK4
{
  double dt;

  template <size_t N>
  Vec<N> operator()
  ( Fn_RHS<N> auto&& rhs,
    const Vec<N>& y
  ) const
  { const Vec<N> k1 = rhs(y           );
    const Vec<N> k2 = rhs(y+(.5*dt)*k1);
    const Vec<N> k3 = rhs(y+(.5*dt)*k2);
    const Vec<N> k4 = rhs(y+(   dt)*k3);
    return y+(dt/6)*(k1+2*k2+2*k3+k4);
  }
};

// `Stepper`: Implicit midpoint rule (Bacchini et al. 2018, Eq. 11).
// Solved by fixed-point iteration: y_next^(k+1) = y + dt*f((y + y_next^(k))/2).
// The metric is stationary so f has no explicit time dependence and the
// midpoint time argument (t^n + t^{n+1})/2 drops out.
struct IMR
{
  double dt;
  int n_iter;

  template <size_t N>
  Vec<N> operator()
  ( Fn_RHS<N> auto&& rhs,
    const Vec<N>& y
  ) const
  { Vec<N> y_next = y;
    for (int k = 0; k < n_iter; k++)
      y_next = y+dt*rhs((y+y_next)*.5);
    return y_next;
  }
};

} // namespace solve
