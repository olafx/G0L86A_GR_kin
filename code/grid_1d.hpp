#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

#include "util.hpp"
#include "common.hpp"
#include "metric.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace maxwell_1d
{

// Schwarzschild radial line through theta=pi/2, phi=0, which is asymptotically
// the +x-axis.

// TODO: We need a Schwarzschild metric to more properly deal with the
//   simplifications this brings.

struct SchwarzschildBLRadialGrid
{

// When considering some 1+1 reduction from the 3+1 metric, different 1+1
// reductions are associated to cells with different volumes, which must be
// accounted for. (In particular for solving Maxwell's equations, here the cell
// volume elements are important.)

  struct MetricReduction
  { double w_vol;
    double norm2_th;
    double norm2_phi;
  };

// TODO: It's weird that these are objects, need to reconsider this. But it's
//   useful that they can be passed as types. We also actually construct these
//   types and pass them that way.

  struct SphericalShellReduction
  {
    [[nodiscard]] MetricReduction operator()
    ( const metric::MetricADMGeoLorentz& m
    ) const
    { return
      { .w_vol = m.lorentz.sqrt_gamma,
        .norm2_th = m.lorentz.gamma_cov.th.th,
        .norm2_phi = m.lorentz.gamma_cov.phi.phi,
      };
    }
  };

  struct ThinFluxTubeReduction
  {
    [[nodiscard]] MetricReduction operator()
    ( const metric::MetricADMGeoLorentz& m
    ) const
    { return
      { .w_vol = sqrt(m.lorentz.gamma_cov.r.r),
        .norm2_th = 1,
        .norm2_phi = 1,
      };
    }
  };

// TODO: Storing the full `Vec3` coordinate is not efficient for this case.
// TODO: Storing this full metric is not efficient for this case.
// TODO: Storing the reduction may not be necessary.
// TODO: May be better to store `cells` and `faces` together.
  struct MetricPoint
  { Vec3_sph x;
    metric::MetricADMGeoLorentz m;
    MetricReduction reduction;
  };

  metric::Kerr::Params metric_params;
  double r_min;
  double r_max;
  double dr;
  size_t n_cells;
  size_t n_faces;
  std::unique_ptr<MetricPoint[]> cells;
  std::unique_ptr<MetricPoint[]> faces;
};

////////////////////////////////////////////////////////////////////////////////

template <typename MetricReduction>
[[nodiscard]] SchwarzschildBLRadialGrid::MetricPoint metric_point
( const metric::Kerr::BoyerLindquist& Schwarzschild_BL,
  double r
)
{ constexpr MetricReduction reduction {};
  const Vec3_sph x {r, std::numbers::pi/2, 0};
  const auto m = Schwarzschild_BL.template eval<true>(x);
  return
  { .x = x,
    .m = m,
    .reduction = reduction(m),
  };
}

// TODO: This could be the `SchwarzschildBLRadialGrid` constructor.

template <typename MetricReduction>
[[nodiscard]] SchwarzschildBLRadialGrid make_grid
( const metric::Kerr::Params& metric_params,
  size_t n_cells,
  double r_min,
  double r_max
)
{ const metric::Kerr::BoyerLindquist Schwarzschild_BL
  { .params = metric_params
  };
  SchwarzschildBLRadialGrid grid
  { .metric_params = metric_params,
    .r_min = r_min,
    .r_max = r_max,
    .dr = (r_max-r_min)/n_cells,
    .n_cells = n_cells,
    .n_faces = n_cells+1,
  };
  grid.cells = std::make_unique_for_overwrite<
    SchwarzschildBLRadialGrid::MetricPoint[]>(grid.n_cells);
  grid.faces = std::make_unique_for_overwrite<
    SchwarzschildBLRadialGrid::MetricPoint[]>(grid.n_faces);

  for (size_t i = 0; i < grid.n_cells; i++)
  { const double r = r_min+(i+.5)*grid.dr;
    grid.cells[i] = metric_point<MetricReduction>(Schwarzschild_BL, r);
  }
  for (size_t i = 0; i < grid.n_faces; i++)
  { const double r = r_min+i*grid.dr;
    grid.faces[i] = metric_point<MetricReduction>(Schwarzschild_BL, r);
  }

  return grid;
}

////////////////////////////////////////////////////////////////////////////////

// Max speed of light in 'radial' Schwarzschild coordinates. Needed for managing
// the CFL condition.
// This is alpha/sqrt(gamma_rr), or for any metric reduction also
// alpha*sqrt(norm2_th*norm2_phi)/w.

[[nodiscard]] double max_radial_light_speed
( const SchwarzschildBLRadialGrid& grid
)
{ double out = 0;
  for (size_t i = 0; i < grid.n_cells; i++)
  { const auto& cell = grid.cells[i];
    out = std::max(out, cell.m.geo.alpha/sqrt(cell.m.geo.gamma_con.r.r));
  }
  for (size_t i = 0; i < grid.n_faces; i++)
  { const auto& face = grid.faces[i];
    out = std::max(out, face.m.geo.alpha/sqrt(face.m.geo.gamma_con.r.r));
  }
  return out;
}

} // namespace maxwell_1d
