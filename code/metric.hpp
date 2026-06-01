#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace metric
{

// We differentiate between spacetimes (like the Kerr spacetime) and metrics
// (like the Boyer-Lindquist coordinates for the Kerr spacetime), since we may
// be interested in different spacetimes and different metrics within the
// spacetime.

// TODO: If we have analytical derivatives, metric derivatives will be computed
//   together with the metric. So these datastructures will need to change, in
//   particular MetricADMGeoLorentz.

// The part of the ADM metric needed to solve a geodesic.
  struct MetricADMGeo
  { double alpha;
    Vec3 beta_con;
    Mat3 gamma_con;
  };
  struct MetricADMDerivativesGeo
  { Vec3 d_alpha;
    Mat3 d_beta_con;
    Ten3 d_gamma_con;
  };

// The additional part of the ADM metric needed for the Lorentz force.
  struct MetricADMLorentz
  { Mat3 gamma_cov;
// det(gamma_ij)^1/2
    double sqrt_gamma;
  };

// A merged metric.
  struct MetricADMGeoLorentz
  { MetricADMGeo geo;
    MetricADMLorentz lorentz;
  };

////////////////////////////////////////////////////////////////////////////////

namespace Kerr
{

// The Kerr spaceetime.

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

////////////////////////////////////////////////////////////////////////////////

// The Boyer-Lindquist coordinates.

struct BoyerLindquist
{
  Params params;

  template <bool with_Lorentz>
  [[nodiscard]] auto eval
  ( Vec3 x
  ) const
  { const double r = x.r;
    const double th = x.th;
    const double r2 = pow(r, 2);
    const double a = params.a;
    const double a2 = pow(a, 2);
    const double sin2_th = pow(sin(th), 2);
    const double cos2_th = pow(cos(th), 2);
    const double M = params.M;
    const double Sig = r2+a2*cos2_th;
    const double Del = r2-2*M*r+a2;
    const double den = (r2+a2)*Sig+2*M*a2*r*sin2_th;

    MetricADMGeo m_geo {};
    m_geo.gamma_con[0][0] = Del/Sig;
    m_geo.gamma_con[1][1] = 1/Sig;
    m_geo.gamma_con[2][2] =
      1/(sin2_th*((r2+a2)+2*M*a2*r*sin2_th/Sig));
    m_geo.beta_con.phi = -2*M*a*r/den;
// NOTE: If ever here this alpha^2 is negative, this will blow up.
    m_geo.alpha = sqrt(Del*Sig/den);

    if constexpr (with_Lorentz)
    { MetricADMLorentz m_Lorentz {};
      m_Lorentz.gamma_cov[0][0] = Sig/Del;
      m_Lorentz.gamma_cov[1][1] = Sig;
      m_Lorentz.gamma_cov[2][2] =
        sin2_th*((r2+a2)+2*M*a2*r*sin2_th/Sig);
      m_Lorentz.sqrt_gamma = Sig*sqrt(sin2_th);
      return MetricADMGeoLorentz {m_geo, m_Lorentz};
    }
    else
      return m_geo;
  }

// TODO: Implementing analytical derivatives.
  [[nodiscard]] MetricADMDerivativesGeo eval_derivatives
  ( Vec3 x
  ) const
  { std::unreachable();
  }

  [[nodiscard]] MetricADMDerivativesGeo eval_derivatives_numerical
  ( const finite_difference::Policy<3> auto& policy_fd,
    Vec3 x
  ) const
  { MetricADMDerivativesGeo out;
// Consider derivatives to some spatial axis i.
    for (size_t i = 0; i < 3; i++)
    {
// The stencil is reused. This central difference stencil requires 2 function
// evaluations. It is abstracted away in finite_difference.hpp.
      const auto stencil = finite_difference::stencils::first_central(
        policy_fd, x, i);
      const Mat23& xs = stencil.xs;
      const Vec2& ws = stencil.ws;
      util::Vec<MetricADMGeo, 2> m
      { eval<false>(xs[0]),
        eval<false>(xs[1])
      };
// The weights and function evaluations we apply ourselves, computing:
// - partial_i alpha
// - partial_i beta^j
// - partial_i gamma^jk
      out.d_alpha[i] = ws[0]*m[0].alpha
                      +ws[1]*m[1].alpha;
      for (size_t j = 0; j < 3; j++)
      { out.d_beta_con[i][j] =
          ws[0]*m[0].beta_con[j]
         +ws[1]*m[1].beta_con[j];
        for (size_t k = 0; k < 3; k++)
          out.d_gamma_con[i][j][k] =
            ws[0]*m[0].gamma_con[j][k]
           +ws[1]*m[1].gamma_con[j][k];
      }
    }
    return out;
  }
};

////////////////////////////////////////////////////////////////////////////////

// TODO: Implementing this.
struct KerrSchildCartesian
{
};

} // namespace Kerr
} // namespace metric
