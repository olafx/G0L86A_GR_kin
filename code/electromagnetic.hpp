#pragma once

#include <cmath>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "util.hpp"
#include "common.hpp"
#include "metric.hpp"

namespace em
{

struct Fields
{ Vec3 D;
  Vec3 B;
};

template <typename Fn>
concept EMField =
  std::invocable<const Fn&, const Vec3&> &&
  std::same_as<std::invoke_result_t<const Fn&, const Vec3&>, Fields>;

struct Vacuum
{ Fields operator()(const Vec3&) const { return {}; }
};

////////////////////////////////////////////////////////////////////////////////

struct Wald
{
  double B_z;
  double B_x;
  double Q;
  metric::Kerr::Params params;

  [[nodiscard]] double psi(double r, double phi) const
  { const double rp = params.r_horizon.p;
    const double rm = params.r_horizon.m;
    return phi + params.a / (rp - rm) * log((r - rp) / (r - rm));
  }

  [[nodiscard]] Fields operator()(const Vec3& x) const
  {
    const double r   = x.r;
    const double th  = x.th;
    const double phi = x.phi;

    const double M = params.M;
    const double a = params.a;

    const double r2 = r*r;
    const double a2 = a*a;

    const double sin_th  = sin(th);
    const double cos_th  = cos(th);
    const double sin2_th = sin_th*sin_th;
    const double cos2_th = cos_th*cos_th;

    const double Sig = r2 + a2*cos2_th;
    const double Del = r2 - 2*M*r + a2;

    const double ps = psi(r, phi);
    const double sin_ps = sin(ps);
    const double cos_ps = cos(ps);

    // psi derivatives
    const double rp = params.r_horizon.p;
    const double rm = params.r_horizon.m;

    const double dpsi_dr =
      a / (rp - rm) * (1.0/(r - rp) - 1.0/(r - rm));

    // =========================
    // F_i0 = ∂_i A_0
    // =========================

    Vec3 F_i0 {};

    // --- F_r0 ---
    {
      double dSig_dr = 2*r;

      double term1 =
        -a*M*B_z*(1+cos2_th) * dSig_dr / (Sig*Sig);

      double term2 =
        a*M*B_x*sin_th*cos_th *
        (
          (cos_ps + r*(-sin_ps*dpsi_dr)) / Sig
          - (r*cos_ps - a*sin_ps)*dSig_dr/(Sig*Sig)
          - a*cos_ps*dpsi_dr/Sig
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
          (cos2_th - sin2_th)*(r*cos_ps - a*sin_ps)/Sig
          - sin_th*cos_th*(r*cos_ps - a*sin_ps)*dSig_dth/(Sig*Sig)
        );

      double term3 =
        r*Q*dSig_dth/(Sig*Sig);

      F_i0[1] = term1 + term2 + term3;
    }

    // --- F_phi0 ---
    {
      F_i0[2] =
        a*M*B_x*sin_th*cos_th/Sig *
        (-r*sin_ps - a*cos_ps);
    }

    // =========================
    // Magnetic tensor F_ij
    // =========================

    // F_rθ
    double F_01;
    {
      double dAr_dth =
        -B_x*(r-M)*(cos2_th - sin2_th)*sin_ps;

      double dAth_dr =
        -a*B_x*sin2_th*cos_ps
        + a*B_x*(r*sin2_th + M*cos2_th)*sin_ps*dpsi_dr;

      F_01 = dAth_dr - dAr_dth;
    }

    // F_rφ
    double F_02;
    {
      double dAr_dphi =
        -B_x*(r-M)*cos_th*sin_th*cos_ps;

      double dAphi_dr =
        B_z*sin2_th*(r - a2*M*(1+cos2_th)/Sig
        + a2*r*M*(1+cos2_th)*2*r/(Sig*Sig))

        - B_x*sin_th*cos_th *
        (
          (2*r - 2*M)*cos_ps
          - Del*sin_ps*dpsi_dr
        );

      F_02 = dAphi_dr - dAr_dphi;
    }

    // F_θφ
    double F_12;
    {
      double dAth_dphi =
        -a*B_x*(r*sin2_th + M*cos2_th)*(-sin_ps)
        -B_x*(r2*cos2_th - r*M*cos(2*th) + a2*cos2_th)*cos_ps;

      double dAphi_dth =
        2*sin_th*cos_th*B_z*
        ((r2+a2)/2 - a2*r*M*(1+cos2_th)/Sig)

        - B_x*(cos2_th - sin2_th)*
        (
          Del*cos_ps
          + (r2+a2)*M/Sig*(r*cos_ps - a*sin_ps)
        );

      F_12 = dAphi_dth - dAth_dphi;
    }

    // =========================
    // Convert to D, B
    // =========================

    const auto f_metric = metric::Kerr::BoyerLindquist{params}.fields(x);

    Fields out {};

    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; j++)
        out.D[i] += f_metric.alpha * f_metric.gamma_con[i][j] * F_i0[j];

    const double inv_sq = 1.0 / f_metric.sqrt_gamma;

    out.B[0] =  F_12 * inv_sq;
    out.B[1] = -F_02 * inv_sq;
    out.B[2] =  F_01 * inv_sq;

    return out;
  }
};

////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] inline std::function<Fields(const Vec3&)> make_field
( const std::string& name,
  const metric::Kerr::Params& params,
  double B_z, double B_x, double Q
)
{
  if (name == "vacuum")
    return Vacuum{};

  if (name == "wald")
    return Wald{B_z, B_x, Q, params};

  throw std::invalid_argument("unknown EM field: " + name);
}

} // namespace em