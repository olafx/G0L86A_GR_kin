#pragma once

#include "Maxwell_1d.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace maxwell_free_1d
{
using namespace maxwell_1d;

// Outgoing wave boundary condition.

struct OutgoingTransverseBoundary
{
  void operator()
  ( const SchwarzschildBLRadialGrid& grid,
    MaxwellYeeGrid& fields
  ) const
  { const size_t n = grid.n_cells;
    const auto C_D_th  = [](const SchwarzschildBLRadialGrid::MetricPoint& m)
    { return m.m.geo.alpha*m.reduction.norm2_th; };
    const auto C_D_phi = [](const SchwarzschildBLRadialGrid::MetricPoint& m)
    { return m.m.geo.alpha*m.reduction.norm2_phi; };
    const auto C_B_th  = [](const SchwarzschildBLRadialGrid::MetricPoint& m)
    { return m.m.geo.alpha*m.reduction.norm2_th; };
    const auto C_B_phi = [](const SchwarzschildBLRadialGrid::MetricPoint& m)
    { return m.m.geo.alpha*m.reduction.norm2_phi; };

    fields.faces[0].B_phi =
      -C_D_th(grid.cells[0  ])/C_B_phi(grid.faces[0])*fields.cells[0  ].D_th;
    fields.faces[n].B_phi =
      +C_D_th(grid.cells[n-1])/C_B_phi(grid.faces[n])*fields.cells[n-1].D_th;
    fields.faces[0].B_th =
      +C_D_phi(grid.cells[0  ])/C_B_th(grid.faces[0])*fields.cells[0  ].D_phi;
    fields.faces[n].B_th =
      -C_D_phi(grid.cells[n-1])/C_B_th(grid.faces[n])*fields.cells[n-1].D_phi;
  }
};

////////////////////////////////////////////////////////////////////////////////

// A step for the 1D-3V Maxwell's equations, for the reduced 1+1 spacetime.

template <typename Boundary>
void step_Maxwell_vac
( const SchwarzschildBLRadialGrid& grid,
  MaxwellYeeGrid& fields,
  double dt,
  const Boundary& boundary
)
{ const auto step_Faraday = [&]
  { for (size_t i = 1; i < grid.n_cells; i++)
    { const double E_phi_R = E_phi(grid.cells[i],   fields.cells[i].D_phi);
      const double E_phi_L = E_phi(grid.cells[i-1], fields.cells[i-1].D_phi);
      fields.faces[i].B_th +=
        dt/grid.faces[i].reduction.w_vol*(E_phi_R-E_phi_L)/grid.dr;
      const double E_th_R = E_th(grid.cells[i],   fields.cells[i].D_th);
      const double E_th_L = E_th(grid.cells[i-1], fields.cells[i-1].D_th);
      fields.faces[i].B_phi -=
        dt/grid.faces[i].reduction.w_vol*(E_th_R-E_th_L)/grid.dr;
    }
  };
  const auto step_Ampere = [&]
  { for (size_t i = 0; i < grid.n_cells; i++)
    { const double H_phi_R = H_phi(grid.faces[i+1], fields.faces[i+1].B_phi);
      const double H_phi_L = H_phi(grid.faces[i],   fields.faces[i].B_phi);
      fields.cells[i].D_th -=
        dt/grid.cells[i].reduction.w_vol*(H_phi_R-H_phi_L)/grid.dr;
      const double H_th_R = H_th(grid.faces[i+1], fields.faces[i+1].B_th);
      const double H_th_L = H_th(grid.faces[i],   fields.faces[i].B_th);
      fields.cells[i].D_phi +=
        dt/grid.cells[i].reduction.w_vol*(H_th_R-H_th_L)/grid.dr;
    }
  };
  step_Faraday();
  boundary(grid, fields);
  step_Ampere();
  boundary(grid, fields);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace maxwell_free_1d
