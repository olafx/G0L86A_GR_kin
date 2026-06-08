#pragma once

#include <numbers>

#include "pic_1d.hpp"
#include "Maxwell_1d.hpp"
#include "geodesic.hpp"
#include "solve.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace vlassov_poisson_1d
{

using Grid = maxwell_1d::SchwarzschildBLRadialGrid;
using Fields = maxwell_1d::MaxwellYeeGrid;

// Integrating Gauss's law.
//
// NOTE: The reduction is significant here, affects the cell volumes and thereby
//   the fluxes.
void solve_Gauss
( const Grid& grid,
  Fields& fields,
  const double* densitized_rho,
  double D_r_flux_left
)
{ double D_r_flux = D_r_flux_left;
  fields.faces[0].D_r = D_r_flux/grid.faces[0].reduction.w_vol;
  for (size_t i = 0; i < grid.n_cells; i++)
  { D_r_flux += 4*std::numbers::pi*densitized_rho[i]*grid.dr;
    fields.faces[i+1].D_r = D_r_flux/grid.faces[i+1].reduction.w_vol;
  }
}

////////////////////////////////////////////////////////////////////////////////

// An electromagnetic field description, to hook up to the Lorentz force.
struct RadialElectricField
{
  const Grid& grid;
  const Fields& fields;

  [[nodiscard]] em::Fields operator()
  ( const metric::MetricADMGeoLorentz&,
    const Vec3_sph& x
  ) const
  { em::Fields out {};
    out.D.r = pic_1d::Bspline<0>::gather_from_faces(grid, x.r,
      [&](size_t i) { return fields.faces[i].D_r; });
    return out;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Deposit charge and solve the electric field.
// 
// NOTE: This is separated because it's useful for diagnostics and such, make
//   sure what we measure is consistent. But it is also a part of course of the
//   PIC step.
template <typename Particle, typename Keep>
void deposit_charge_and_solve_Gauss
( const Grid& grid,
  Fields& fields,
  const Particle* particles,
  size_t n_particles,
  double* densitized_rho,
  const Keep& keep,
  double D_r_flux_left
)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < grid.n_cells; i++)
    densitized_rho[i] = 0;
  pic_1d::add_densitized_charge(
    grid, particles, n_particles, densitized_rho, keep);
  solve_Gauss(grid, fields, densitized_rho, D_r_flux_left);
}

// 1D-1V Vlasov-Poisson PIC step.
// Particles at [r]^n, [u_r]^(n-1/2).
//
// Step 1: Deposit charge at [r]^n.
// Step 2: Solve [D^r]^n.
// Step 3: [u_r]^(n-1/2) kick.
// Step 4: [r]^n drift.
template <typename Particle, typename Stepper, typename Boundary>
void step
( const Grid& grid,
  Fields& fields,
  Particle* particles,
  size_t n_particles,
  double* densitized_rho,
  const Stepper& solver,
  const finite_difference::policies::Simple& policy_fd,
  const Boundary& boundary
)
{ deposit_charge_and_solve_Gauss(
    grid, fields, particles, n_particles, densitized_rho,
    [](const Particle& p) { return p.mobile; }, 0);
  const metric::Kerr::BoyerLindquist metric
  { .params = grid.metric_params };
  const RadialElectricField em_field
  { .grid = grid, .fields = fields };

// TODO: Dynamic scheduling may be better as particle become immobile.
#pragma omp parallel for schedule(static)
  for (size_t i_p = 0; i_p < n_particles; i_p++)
  { auto& p = particles[i_p];
    if (!p.mobile)
      continue;
    const geodesic::Problem problem
    { metric, em_field, policy_fd, p.species };
    const auto rhs_u = [&](const Vec3& u)
    { return problem.rhs_u(p.x, u); };
    p.u = solver(rhs_u, p.u);
    p.u.th = 0;
    p.u.phi = 0;
    const auto rhs_x = [&](const Vec3& x)
    { return problem.rhs_x(x, p.u); };
    p.x = solver(rhs_x, p.x);
    p.x.th = std::numbers::pi/2;
    p.x.phi = 0;
    p.r_prev = p.x.r;
    boundary(grid, p);
  }
}

} // namespace vlassov_poisson_1d
