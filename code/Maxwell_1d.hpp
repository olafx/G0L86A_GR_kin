#pragma once

#include <cmath>
#include <memory>

#include "grid_1d.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace maxwell_1d
{

// 1D-3V Yee lattice
//
// Cell centers:
//   D^theta, D^phi, B^r
// Faces:
//   D^r, B^theta, B^phi

// TODO: We don't always need 1D-3V, the electrostatic simulation is 1D-3V, and
//   only needs D_r really.
// TODO: May be better to store `cells` and `faces` together.

struct MaxwellYeeGrid
{
  struct Cell
  { double D_th;
    double D_phi;
    double B_r;
  };

  struct Face
  { double D_r;
    double B_th;
    double B_phi;
  };

  size_t n_cells;
  size_t n_faces;
  std::unique_ptr<Cell[]> cells;
  std::unique_ptr<Face[]> faces;
};

// TODO: This could be the `MaxwellYeeGrid` constructor.

[[nodiscard]] MaxwellYeeGrid make_fields
( const SchwarzschildBLRadialGrid& grid
)
{ MaxwellYeeGrid out
  { .n_cells = grid.n_cells,
    .n_faces = grid.n_faces,
  };
  out.cells = std::make_unique<MaxwellYeeGrid::Cell[]>(out.n_cells);
  out.faces = std::make_unique<MaxwellYeeGrid::Face[]>(out.n_faces);
  return out;
}

////////////////////////////////////////////////////////////////////////////////

// Derived covariant electromagnetic fields.
// 
// NOTE: This makes the zero-shift assumption (beta^i=0), which is generally the
//   case for a Boyer-Lindquist metric. Without this assumption the electric and
//   magnetic intrinsic and derived fields are coupled.

// TODO: We should not be making this assumption here without this being
//   reflected some way in the structure of the code. Like this should be a
//   special kind of simplification associated to some type. It doesn't make
//   sense to begin with that this is specific to `SchwarzschildBLRadialGrid`.

[[nodiscard]] double E_r
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_r
)
{ return p.m.geo.alpha*p.m.lorentz.gamma_cov.r.r*D_r;
}

[[nodiscard]] double E_th
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_th
)
{ return p.m.geo.alpha*p.reduction.norm2_th*D_th;
}

[[nodiscard]] double E_phi
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_phi
)
{ return p.m.geo.alpha*p.reduction.norm2_phi*D_phi;
}

[[nodiscard]] double H_r
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_r
)
{ return p.m.geo.alpha*p.m.lorentz.gamma_cov.r.r*B_r;
}

[[nodiscard]] double H_th
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_th
)
{ return p.m.geo.alpha*p.reduction.norm2_th*B_th;
}

[[nodiscard]] double H_phi
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_phi
)
{ return p.m.geo.alpha*p.reduction.norm2_phi*B_phi;
}

////////////////////////////////////////////////////////////////////////////////

// Locally measured quantities for the 1+1 setup.

[[nodiscard]] double local_E_r
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_r
)
{ return E_r(p, D_r)/sqrt(p.m.lorentz.gamma_cov.r.r);
}

[[nodiscard]] double local_E_th
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_th
)
{ return E_th(p, D_th)/sqrt(p.reduction.norm2_th);
}

[[nodiscard]] double local_E_phi
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double D_phi
)
{ return E_phi(p, D_phi)/sqrt(p.reduction.norm2_phi);
}

[[nodiscard]] double local_B_r
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_r
)
{ return H_r(p, B_r)/sqrt(p.m.lorentz.gamma_cov.r.r);
}

[[nodiscard]] double local_B_th
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_th
)
{ return H_th(p, B_th)/sqrt(p.reduction.norm2_th);
}

[[nodiscard]] double local_B_phi
( const SchwarzschildBLRadialGrid::MetricPoint& p,
  double B_phi
)
{ return H_phi(p, B_phi)/sqrt(p.reduction.norm2_phi);
}

} // namespace maxwell_1d
