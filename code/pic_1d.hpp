#pragma once

#include <cmath>

#include "grid_1d.hpp"
#include "geodesic.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pic_1d
{

// 1D-1V and 1D-3V PIC, for the radial Schwarzschild setup.

using Grid = maxwell_1d::SchwarzschildBLRadialGrid;

// TODO: Inefficient data structure. For example, `x`, `u`, `species`.

template <typename Shape_>
struct Particle
{ using Shape = Shape_;
  Vec3_sph x;
  Vec3 u;
  geodesic::particle::Charged species;
  double r_prev;
  double m;
  int charge_sign;
  int label;
  bool mobile;
};

////////////////////////////////////////////////////////////////////////////////

template <size_t Order>
struct Bspline;

template <>
struct Bspline<0>
{
  template <typename Particle, typename F>
  static void deposit_to_cells
  ( const Grid& grid,
    const Particle& p,
    const F& f
  )
  { const double z = (p.x.r-grid.r_min)/grid.dr;
    const long i = floor(z);
    f(i);
  }

  template <typename F>
  [[nodiscard]] static double gather_from_faces
  ( const Grid& grid,
    double r,
    const F& f
  )
  { const double z = (r-grid.r_min)/grid.dr;
    const long i = lround(z);
    return f(i);
  }

  template <typename F>
  [[nodiscard]] static double gather_from_cells
  ( const Grid& grid,
    double r,
    const F& f
  )
  { const double z = (r-grid.r_min)/grid.dr;
    const long i = floor(z);
    return f(i);
  }
};

////////////////////////////////////////////////////////////////////////////////

// TODO: If we parallelize over the particles, there is a race condition
//   updating `densitized_rho`. An atomic update here is an ugly solution
//   because those locations are predictable and there are few of them.

template <typename Particle, typename Keep>
void add_densitized_charge
( const Grid& grid,
  const Particle* particles,
  size_t n_particles,
  double* densitized_rho,
  const Keep& keep
)
{
// TODO: Dynamic scheduling may be better as particle become immobile.
#pragma omp parallel for schedule(static)
  for (size_t i_p = 0; i_p < n_particles; i_p++)
  { const Particle& p = particles[i_p];
    if (!keep(p))
      continue;
    const double rho_p = p.m*p.species.q_over_m/grid.dr;
    Particle::Shape::deposit_to_cells(grid, p,
      [&](size_t i)
      {
#pragma omp atomic update
        densitized_rho[i] += rho_p;
      });
  }
}

////////////////////////////////////////////////////////////////////////////////

template <typename Particle>
void wrap_to_domain
( double r_min,
  double L,
  Particle& p
)
{ p.x.r = r_min+fmod(p.x.r-r_min, L);
  if (p.x.r < r_min)
    p.x.r += L;
}

// Local 3-velocity to covariant u_i, useful for initialization.
// 
// NOTE: This is correct for any metric.
// TODO: This is in a questionable location. It probably more accurately
//   belongs to `metric.hpp`.
[[nodiscard]] double radial_local_v_to_cov_u
( const metric::MetricADMGeoLorentz& m,
  double v_r
)
{ const double gamma = 1/sqrt(1-v_r*v_r);
  return sqrt(m.lorentz.gamma_cov.r.r)*gamma*v_r;
}

} // namespace pic_1d
