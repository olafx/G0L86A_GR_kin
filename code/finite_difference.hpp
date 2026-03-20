#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////

namespace finite_difference
{

template <size_t N>
using Vec = util::Vec<double, N>;

template <size_t N1, size_t N2>
using Mat = util::Ten<double, N1, N2>::V;

// We overcomplicate picking a step h a bit here if in the future there are
// additional constraints.

template <typename Fn, size_t N>
concept StepPolicy =
  std::invocable<const Fn&, const Vec<N>&, size_t> &&
  std::same_as<
    std::invoke_result_t<const Fn&, const Vec<N>&, size_t>,
    double>;

// A simple StepPolicy with some minimum h and minimum absolute h.
// 
// NOTE: Using operator() here makes this an invocable, a valid StepPolicy, and
//   it is convenient to keep the policy as a struct.
struct StepPolicy_Simple
{ double h_rel;
  double h_min;

  template <size_t N>
  [[nodiscard]] double operator()
  ( const Vec<N>& x,
    size_t axis
  ) const
  { return std::max(h_min, h_rel*abs(x[axis]));
  }
};

////////////////////////////////////////////////////////////////////////////////

// A Stencil describes how a derivative is approximated. This can represent
// arbitrary derivatives in arbitrary dimensions in principle, as specific
// usages.
template <size_t N, size_t K>
struct Stencil
{ Mat<N, K> xs;
  Vec<K> ws;
};

// 1st derivative, central difference
// 
// f' = (f(x+he)-f(x-he))/2h
// x is a vector, the derivative is computed according to some axis with normal
// vector e.
template <size_t N>
[[nodiscard]] Stencil<N, 2> first_central
( StepPolicy<N> auto&& step_policy,
  const Vec<N>& x,
  size_t axis
)
{ const double h = step_policy(x, axis);
  Stencil<N, 2> out;
  out.xs[0] = x;
  out.xs[1] = x;
  out.xs[0][axis] = x[axis]-h;
  out.xs[1][axis] = x[axis]+h;
  out.ws[0] = -.5/h;
  out.ws[1] = +.5/h;
  return out;
}

// 2D Cartesian Laplacian, central difference
// 
// Lap f = (f(x+hi ei)+f(x-hi ei))/hi^2
//       - 2f(x)/hi^2
//       + (f(x+hj ej)+f(x-hj ej))/hi^2
//       - 2f(x)/hj^2
// 
// NOTE: We don't use this, it's just to understand how this stencil abstraction
//   works.
template <size_t N>
[[nodiscard]] Stencil<2, 5> Laplacian_central
( StepPolicy<2> auto&& step_policy,
  const Vec<2>& x
)
{ const Vec<2> h
  { step_policy(x, 0),
    step_policy(x, 1)
  };
  const Vec<2> h2 = h*h;
  Stencil<N, 5> out;
  for (size_t i = 0; i < 5; i++)
    out.xs[i] = x;
  out.xs[1][0] = x[0]-h.x;
  out.xs[2][0] = x[0]+h.x;
  out.xs[3][1] = x[1]-h.y;
  out.xs[4][1] = x[1]+h.y;
  out.ws[0] = -2/h2.x-2/h2.y;
  out.ws[1] = 1/h2.x;
  out.ws[2] = 1/h2.x;
  out.ws[3] = 1/h2.y;
  out.ws[4] = 1/h2.y;
  return out;
}

} // namespace finite_difference
