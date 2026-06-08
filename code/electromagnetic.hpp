#pragma once

#include "util.hpp"
#include "common.hpp"
#include "finite_difference.hpp"
#include "metric.hpp"

namespace em
{

struct Fields
{ Vec3 D;
  Vec3 B;
};

struct Vacuum
{
  Fields operator()
  ( const metric::MetricADMGeoLorentz&,
    const Vec3&
  ) const
  { return {};
  }
};

////////////////////////////////////////////////////////////////////////////////

// The Wald solution around a Kerr black hole.
//
// NOTE: The Wald solution is dependent on the choice of Kerr spacetime
//   coordinates.

template <typename Metric>
struct Wald;

template <>
struct Wald<metric::Kerr::BoyerLindquist>
{
  struct Params
  { double B_z;
    double B_x;
    double Q;
    double h_rel;
    double h_min;
  };

  Params wald_params;
  const metric::Kerr::Params& metric_params;

  [[nodiscard]] double psi
  ( const Vec3_sph& x
  ) const
  { const double rp = metric_params.r_horizon.p;
    const double rm = metric_params.r_horizon.m;
    return x.phi + metric_params.a / (rp - rm) * log((x.r - rp) / (x.r - rm));
  }

  // The 4-potential A_mu from Bacchini et al. 2019, Eqs. (39)-(42),
  // referencing Kopacek & Karas (2014).
  // A_t, A_r, A_th, A_phi in Boyer-Lindquist coordinates.
 private:
  struct FourPotential { double A_t, A_r, A_th, A_phi; };

  [[nodiscard]] FourPotential A_mu
  ( const Vec3_sph& x
  ) const
  {
    const double r   = x.r;
    const double th  = x.th;

    const double M   = metric_params.M;
    const double a   = metric_params.a;
    const double B_z = wald_params.B_z;
    const double B_x = wald_params.B_x;
    const double Q   = wald_params.Q;

    const double r2 = r*r;
    const double a2 = a*a;

    const double st = sin(th);
    const double ct = cos(th);
    const double s2 = st*st;
    const double c2 = ct*ct;

    const double Sig = r2 + a2*c2;
    const double Del = r2 - 2*M*r + a2;

    const double psi_ = psi(x);
    const double sp = sin(psi_);
    const double cp = cos(psi_);

    FourPotential A;

    // Eq. (39): A_t
    A.A_t  = a * r * M * B_z / Sig * (1.0 + c2) - a * B_z;
    A.A_t += a * M * B_x * st * ct / Sig * (r * cp - a * sp);
    A.A_t -= r * Q / Sig;

    // Eq. (40): A_r
    A.A_r = -B_x * (r - M) * ct * st * sp;

    // Eq. (41): A_th
    A.A_th  = -a * B_x * (r * s2 + M * c2) * cp;
    A.A_th -= B_x * (r2 * c2 - r * M * cos(2.0*th) + a2 * cos(2.0*th)) * sp;

    // Eq. (42): A_phi
    A.A_phi  = B_z * s2 * ((r2 + a2) / 2.0 - a2 * r * M / Sig * (1.0 + c2));
    A.A_phi -= B_x * st * ct * (Del * cp + (r2 + a2) * M / Sig * (r * cp - a * sp));
    A.A_phi += a * r * Q * s2 / Sig;

    return A;
  }

 public:
  // Compute F = dA via 2nd-order central finite differences,
  // using the existing stencil infrastructure from finite_difference.hpp.
  [[nodiscard]] Fields operator()
  ( const metric::MetricADMGeoLorentz& m,
    const Vec3_sph& x
  ) const
  {
    const finite_difference::policies::Simple policy_fd
      {wald_params.h_rel, wald_params.h_min};

    // Build stencils: each is a pair {xs[0], xs[1]} with weights {ws[0], ws[1]}
    // where ws[0] = -1/(2h), ws[1] = +1/(2h) for central difference.
    const auto st_r   = finite_difference::stencils::first_central(policy_fd, x, 0);
    const auto st_th  = finite_difference::stencils::first_central(policy_fd, x, 1);
    const auto st_phi = finite_difference::stencils::first_central(policy_fd, x, 2);

    // Evaluate A_mu at each stencil point.
    const auto Am_r0   = A_mu(st_r.xs[0]);
    const auto Am_r1   = A_mu(st_r.xs[1]);
    const auto Am_th0  = A_mu(st_th.xs[0]);
    const auto Am_th1  = A_mu(st_th.xs[1]);
    const auto Am_phi0 = A_mu(st_phi.xs[0]);
    const auto Am_phi1 = A_mu(st_phi.xs[1]);

    // F_i0 = partial_i A_t
    Vec3 F_i0;
    F_i0[0] = st_r.ws[0]*Am_r0.A_t     + st_r.ws[1]*Am_r1.A_t;
    F_i0[1] = st_th.ws[0]*Am_th0.A_t   + st_th.ws[1]*Am_th1.A_t;
    F_i0[2] = st_phi.ws[0]*Am_phi0.A_t + st_phi.ws[1]*Am_phi1.A_t;

    // F_rth  = partial_r A_th  - partial_th A_r
    const double F_01 =
      (st_r.ws[0]*Am_r0.A_th   + st_r.ws[1]*Am_r1.A_th)
    - (st_th.ws[0]*Am_th0.A_r  + st_th.ws[1]*Am_th1.A_r);

    // F_rphi = partial_r A_phi - partial_phi A_r
    const double F_02 =
      (st_r.ws[0]*Am_r0.A_phi   + st_r.ws[1]*Am_r1.A_phi)
    - (st_phi.ws[0]*Am_phi0.A_r + st_phi.ws[1]*Am_phi1.A_r);

    // F_thphi = partial_th A_phi - partial_phi A_th
    const double F_12 =
      (st_th.ws[0]*Am_th0.A_phi   + st_th.ws[1]*Am_th1.A_phi)
    - (st_phi.ws[0]*Am_phi0.A_th  + st_phi.ws[1]*Am_phi1.A_th);

    // Convert to D, B
    em::Fields out;

    // B 
    out.B[0] =  F_12 / m.lorentz.sqrt_gamma;
    out.B[1] = -F_02 / m.lorentz.sqrt_gamma;
    out.B[2] =  F_01 / m.lorentz.sqrt_gamma;

    // e_{jkl} β^k B^l
    const Vec3 bxB = m.lorentz.sqrt_gamma * cross(m.geo.beta_con, out.B);

    // Eq. (14):  α D^i = γ^{ij}(E_j − (β×B)_j)
    out.D = {0, 0, 0};
    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; j++)
        out.D[i] += m.geo.gamma_con[i][j] * (F_i0[j] - bxB[j]);
    
    out.D[0] /= m.geo.alpha;
    out.D[1] /= m.geo.alpha;
    out.D[2] /= m.geo.alpha;

    return out;
  }
};

} // namespace em
