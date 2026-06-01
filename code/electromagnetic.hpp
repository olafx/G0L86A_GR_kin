#pragma once

#include "util.hpp"
#include "common.hpp"
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

  [[nodiscard]] Fields operator()
  ( const metric::MetricADMGeoLorentz& m,
    const Vec3_sph& x
  ) const
  {
    const double r   = x.r;
    const double th  = x.th;
    const double phi = x.phi;

    const double M   = metric_params.M;
    const double a   = metric_params.a;
    const double B_z = wald_params.B_z;
    const double B_x = wald_params.B_x;
    const double Q   = wald_params.Q;

    const double r2 = r*r;
    const double a2 = a*a;

    const double sin_th  = sin(th);
    const double cos_th  = cos(th);
    const double sin2_th = sin_th*sin_th;
    const double cos2_th = cos_th*cos_th;

    const double Sig = r2 + a2*cos2_th;
    const double Del = r2 - 2*M*r + a2;

    const double psi_ = psi(x);
    const double sin_psi_ = sin(psi_);
    const double cos_psi_ = cos(psi_);

    // psi derivatives
    const double rp = metric_params.r_horizon.p;
    const double rm = metric_params.r_horizon.m;

    const double dpsi_dr =
      a / (rp - rm) * (1.0/(r - rp) - 1.0/(r - rm));

    // =========================
    // F_i0 = partial_i A_0
    // =========================

    Vec3 F_i0;

    // --- F_r0 ---
    {
      double dSig_dr = 2*r;

      double term1 =
        -a*M*B_z*(1+cos2_th) * dSig_dr / (Sig*Sig);

      double term2 =
        a*M*B_x*sin_th*cos_th *
        (
          (cos_psi_ + r*(-sin_psi_*dpsi_dr)) / Sig
          - (r*cos_psi_ - a*sin_psi_)*dSig_dr/(Sig*Sig)
          - a*cos_psi_*dpsi_dr/Sig
        );

      double term3 =
        -Q/Sig + r*Q*dSig_dr/(Sig*Sig);

      F_i0[0] = term1 + term2 + term3;
    }

    // --- F_theta0 ---
    {
      double dSig_dth = -2*a2*cos_th*sin_th;

      double term1 =
        a*M*B_z *
        (
          -2*cos_th*sin_th/Sig
          - (1+cos2_th)*dSig_dth/(Sig*Sig)
        );

      double term2 =
        a*M*B_x *
        (
          (cos2_th - sin2_th)*(r*cos_psi_ - a*sin_psi_)/Sig
          - sin_th*cos_th*(r*cos_psi_ - a*sin_psi_)*dSig_dth/(Sig*Sig)
        );

      double term3 =
        r*Q*dSig_dth/(Sig*Sig);

      F_i0[1] = term1 + term2 + term3;
    }

    // --- F_phi0 ---
    {
      F_i0[2] =
        a*M*B_x*sin_th*cos_th/Sig *
        (-r*sin_psi_ - a*cos_psi_);
    }

    // =========================
    // Magnetic tensor F_ij
    // =========================

    // F_rtheta
    double F_01;
    {
      double dAr_dth =
        -B_x*(r-M)*(cos2_th - sin2_th)*sin_psi_;

      double dAth_dr =
        -a*B_x*sin2_th*cos_psi_
        + a*B_x*(r*sin2_th + M*cos2_th)*sin_psi_*dpsi_dr;

      F_01 = dAth_dr - dAr_dth;
    }

    // F_rphi
    double F_02;
    {
      double dAr_dphi =
        -B_x*(r-M)*cos_th*sin_th*cos_psi_;

      double dAphi_dr =
        B_z*sin2_th*(r - a2*M*(1+cos2_th)/Sig
        + a2*r*M*(1+cos2_th)*2*r/(Sig*Sig))

        - B_x*sin_th*cos_th *
        (
          (2*r - 2*M)*cos_psi_
          - Del*sin_psi_*dpsi_dr
        );

      F_02 = dAphi_dr - dAr_dphi;
    }

    // F_thetaphi
    double F_12;
    {
      double dAth_dphi =
        -a*B_x*(r*sin2_th + M*cos2_th)*(-sin_psi_)
        -B_x*(r2*cos2_th - r*M*cos(2*th) + a2*cos2_th)*cos_psi_;

      double dAphi_dth =
        2*sin_th*cos_th*B_z*
        ((r2+a2)/2 - a2*r*M*(1+cos2_th)/Sig)

        - B_x*(cos2_th - sin2_th)*
        (
          Del*cos_psi_
          + (r2+a2)*M/Sig*(r*cos_psi_ - a*sin_psi_)
        );

      F_12 = dAphi_dth - dAth_dphi;
    }

    // =========================
    // Convert to D, B
    // =========================

    em::Fields out;

    out.D = {0, 0, 0};
    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; j++)
        out.D[i] += m.geo.alpha * m.geo.gamma_con[i][j] * F_i0[j];

    out.B[0] =  F_12/m.lorentz.sqrt_gamma;
    out.B[1] = -F_02/m.lorentz.sqrt_gamma;
    out.B[2] =  F_01/m.lorentz.sqrt_gamma;

    return out;
  }
};

} // namespace em
