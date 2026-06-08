#pragma once

#include <cmath>
#include <numbers>

#include "Maxwell_1d.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace maxwell_1d
{
namespace diagnostics
{

// Electromagnetic field energy.
[[nodiscard]] double energy_EM
( const SchwarzschildBLRadialGrid& grid,
  const MaxwellYeeGrid& fields
)
{ double out = 0;
  for (size_t i = 0; i < grid.n_cells; i++)
  { const auto& c = grid.cells[i];
    const double ED =
      E_th (c, fields.cells[i].D_th )*fields.cells[i].D_th
     +E_phi(c, fields.cells[i].D_phi)*fields.cells[i].D_phi;
    const double HB = H_r(c, fields.cells[i].B_r)*fields.cells[i].B_r;
    out += c.reduction.w_vol*(ED+HB)*grid.dr;
  }
  for (size_t i = 0; i < grid.n_faces; i++)
  { const auto& f = grid.faces[i];
    const double ED = E_r(f, fields.faces[i].D_r)*fields.faces[i].D_r;
    const double HB =
      H_th (f, fields.faces[i].B_th )*fields.faces[i].B_th
     +H_phi(f, fields.faces[i].B_phi)*fields.faces[i].B_phi;
// Boundary faces count for half.
    const double w = (i == 0 || i == grid.n_faces-1) ? .5 : 1;
    out += w*f.reduction.w_vol*(ED+HB)*grid.dr;
  }
  out *= 1./(8*std::numbers::pi);
  return out;
}

// Massive article kinetic energy in the Euclidian convention.
template <typename Particle>
[[nodiscard]] double energy_kinetic
( const SchwarzschildBLRadialGrid& grid,
  const Particle* particles,
  size_t n_particles
)
{ double K = 0;
#pragma omp parallel for schedule(static) reduction(+:K)
  for (size_t i_p = 0; i_p < n_particles; i_p++)
  { const Particle& p = particles[i_p];
    if (!p.mobile)
      continue;
    const size_t i_c = floor((p.x.r-grid.r_min)/grid.dr);
    const auto& c = grid.cells[i_c];
    const auto& gamma_con = c.m.geo.gamma_con;
    const double u2 =
      gamma_con.r.r    *pow(p.u.r, 2)
     +gamma_con.th.th  *pow(p.u.th, 2)
     +gamma_con.phi.phi*pow(p.u.phi, 2);
    const double W = sqrt(p.species.eps+u2);
    K += p.m*(W-1);
  }
  return K;
}

} // namespace diagnostics
} // namespace maxwell_1d
