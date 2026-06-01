#pragma once

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

////////////////////////////////////////////////////////////////////////////////

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
// Iteration stops early when the update norm falls below `tol`, or after
// at most 150 iterations regardless of `n_iter`.
//
// Olaf: I don't understand this max 150 feature regardless of n_iter. Just
//   restrict n_iter to be max 150 beforehand, does the same thing.
struct IMR
{
  double dt;
  double tol;
  size_t n_iter;

  template <size_t N>
  Vec<N> operator()
  ( Fn_RHS<N> auto&& rhs,
    const Vec<N>& y
  ) const
  { Vec<N> y_next = y;
    for (size_t k = 0; k < n_iter; k++)
    { const Vec<N> y_new = y+dt*rhs((y+y_next)*.5);
      if (norm2(y_new-y_next) < tol*tol)
        return y_new;
      y_next = y_new;
    }
    return y_next;
  }
};

} // namespace solve
