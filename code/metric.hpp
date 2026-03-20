#pragma once

#include <cmath>
#include <cstddef>

#include "finite_difference.hpp"
#include "util.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace metric
{

using Vec2 = util::Vec<double, 2>;
using Vec3 = util::Vec<double, 3>;
using Mat3 = util::Ten<double, 3, 3>::V;
using Mat23 = util::Ten<double, 3, 2>::V;
using Ten3 = util::Ten<double, 3, 3, 3>::V;

// We differentiate between spacetimes (like the Kerr spacetime) and metrics
// (like the Boyer-Lindquist coordinates for the Kerr spacetime), since we may
// be interested in different spacetimes and different metrics within the
// spacetime.

namespace Kerr
{

// Precompute some properties of the spacetime.
struct Params
{
  double M;
  double a;
  Vec2 r_horizon; // [inner, outer]

  Params
  ( double M,
    double a
  ) : M(M),
      a(a)
  { r_horizon.m = M-sqrt(M*M-a*a);
    r_horizon.p = M+sqrt(M*M-a*a);
  }
};

struct BoyerLindquist
{
  struct Fields
  { double alpha;
    Vec3 beta_con;
    Mat3 gamma_con;
  };

  struct Derivatives
  { Vec3 d_alpha;
    Mat3 d_beta_con;
    Ten3 d_gamma_con;
  };

  [[nodiscard]] Fields fields
  ( const Params& params_phys,
    Vec3 x
  ) const
  { const double r = x.r;
    const double r2 = pow(r, 2);
    const double a = params_phys.a;
    const double a2 = pow(a, 2);
    const double th = x.th;
    const double sin2_th = pow(sin(th), 2);
    const double cos2_th = pow(cos(th), 2);
    const double M = params_phys.M;
    const double Sig = r2+a2*cos2_th;
    const double Del = r2-2*M*r+a2;
    const double den = (r2+a2)*Sig+2*M*a2*r*sin2_th;

// Default initialize to 0 for convenience.
    Fields out {};
    out.gamma_con[0][0] = Del/Sig;
    out.gamma_con[1][1] = 1/Sig;
    out.gamma_con[2][2] = 1/(sin2_th*((r2+a2)+2*M*a2*r*sin2_th/Sig));
    out.beta_con.phi = -2*M*a*r/den;
// NOTE: If ever here this alpha^2 is negative, this will blow up.
    out.alpha = sqrt(Del*Sig/den);
    return out;
  }

// TODO: Implementing analytical derivatives.
  [[nodiscard]] Derivatives derivatives
  ( const Params& params_phys,
    Vec3 x
  ) const
  {
  }

  [[nodiscard]] Derivatives derivatives_numerical
  ( const Params& params_phys,
    const finite_difference::StepPolicy_Simple& step_policy,
    Vec3 x
  ) const
  { Derivatives out;
// Consider derivatives to some spatial axis i.
    for (size_t i = 0; i < 3; i++)
    {
// The stencil is reused. This central difference stencil requires 2 function
// evaluations. It is abstracted away in finite_difference.hpp.
      const auto stencil = finite_difference::first_central(step_policy, x, i);
      const Mat23& xs = stencil.xs;
      const Vec2& ws = stencil.ws;
      util::Vec<Fields, 2> f
      { fields(params_phys, xs[0]),
        fields(params_phys, xs[1])
      };
// The weights and function evaluations we apply ourselves, computing:
// - partial_i alpha
// - partial_i beta^j
// - partial_i gamma^jk
      out.d_alpha[i] = ws[0]*f[0].alpha
                      +ws[1]*f[1].alpha;
      for (size_t j = 0; j < 3; j++)
      { out.d_beta_con[i][j] =
          ws[0]*f[0].beta_con[j]
         +ws[1]*f[1].beta_con[j];
        for (size_t k = 0; k < 3; k++)
          out.d_gamma_con[i][j][k] =
            ws[0]*f[0].gamma_con[j][k]
           +ws[1]*f[1].gamma_con[j][k];
      }
    }
    return out;
  }
};

// TODO: Implementing this.
struct KerrSchildCartesian
{

};

} // namespace Kerr
} // namespace metric
