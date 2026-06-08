#include <algorithm>
#include <cmath>
#include <mdspan>
#include <vector>

#include "util.hpp"
#include "common.hpp"
#include "Maxwell_free_1d.hpp"
#include "diagnostics_1d.hpp"
#include "pic_1d.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace
{

namespace py = pybind11;
using Grid = maxwell_1d::SchwarzschildBLRadialGrid;

////////////////////////////////////////////////////////////////////////////////

// TODO: This needs to just come from the metric, but we don't have a nice
//   Schwarzschild metric yet.
[[nodiscard]] double Schwarzschild_lapse
( double M,
  double r
)
{ return sqrt(1-2*M/r);
}

// TODO: This needs to come from the metric, but we don't have a nice
//   Schwarzschild metric yet.
[[nodiscard]] double Schwarzschild_tortoise
( double M,
  double r
)
{ const double r_horizon = 2*M;
  return r+r_horizon*log(r/r_horizon-1);
}

// The expected redshift ratio is alpha(emit)/alpha(obs).
[[nodiscard]] double redshift_ratio
( double M,
  double r_emit,
  double r_obs
)
{ return Schwarzschild_lapse(M, r_emit)/Schwarzschild_lapse(M, r_obs);
}

////////////////////////////////////////////////////////////////////////////////

// Initialize an outgoing wave along +x, via D^theta and B^phi components,
// according to some profile. A Tortoise profile will be used.

template <typename Profile>
void initialize_fields
( const maxwell_free_1d::SchwarzschildBLRadialGrid& grid,
  maxwell_free_1d::MaxwellYeeGrid& fields,
  const Profile& profile,
  double dt
)
{ for (size_t i = 0; i < grid.n_cells; i++)
  { const auto& c = grid.cells[i];
    fields.cells[i].D_th =
      profile(c.x.r, 0)/(c.m.geo.alpha*c.reduction.norm2_th);
    fields.cells[i].D_phi = 0;
    fields.cells[i].B_r = 0;
  }
  for (size_t i = 0; i < grid.n_faces; i++)
  { const auto& f = grid.faces[i];
    fields.faces[i].D_r = 0;
    fields.faces[i].B_th = 0;
    fields.faces[i].B_phi =
      profile(f.x.r, -.5*dt)/(f.m.geo.alpha*f.reduction.norm2_phi);
  }
}

struct TortoiseProfile
{
  double M;
  double r_center;
  double width;
  double omega;
  double amplitude;

  [[nodiscard]] double operator()
  ( double r,
    double t
  ) const
  { const double r_star = M == 0 ? r : Schwarzschild_tortoise(M, r);
    const double r_center_star = Schwarzschild_tortoise(M, r_center);
    const double phase = r_star-r_center_star-t;
    const double envelope = exp(-.5*phase*phase/pow(width, 2));
    return amplitude*envelope*cos(omega*phase);
  }
};

////////////////////////////////////////////////////////////////////////////////

struct Diagnostics
{ double time;
  size_t step;
  double dt;
  double dr;
  double energy_EM;
  double max_abs_D_theta;
  double max_abs_D_phi;
  double max_abs_B_theta;
  double max_abs_B_phi;
};

struct Simulation
{
  static constexpr maxwell_free_1d::OutgoingTransverseBoundary boundary {};
  double M;
  Grid grid;
  maxwell_free_1d::MaxwellYeeGrid fields;
  double dt;
  double time;
  size_t step;
  std::vector<double> probe_radii;
  std::vector<double> probe_time;
  std::vector<double> probe_proper_time;
  std::vector<double> probe_E_theta;

  Simulation
  ( double M,
    double r_min,
    double r_max,
    size_t n_cells,
    double cfl,
    std::vector<double> probe_radii
  ) : M(M),
      grid(maxwell_free_1d::make_grid
        <maxwell_1d::SchwarzschildBLRadialGrid::ThinFluxTubeReduction>(
        metric::Kerr::Params {M, 0}, n_cells, r_min, r_max)),
      fields(
        maxwell_free_1d::make_fields(grid)),
      dt(
        cfl*grid.dr/maxwell_free_1d::max_radial_light_speed(grid)),
      probe_radii(
        std::move(probe_radii)),
      time(0),
      step(0)
  {}

  void initialize
  ( double r_center,
    double width,
    double omega
  )
  { const double amplitude = 1;
    const TortoiseProfile profile
    { .M = M,
      .r_center = r_center,
      .width = width,
      .omega = omega,
      .amplitude = amplitude,
    };
    initialize_fields(grid, fields, profile, dt);
    boundary(grid, fields);
    clear_probe_history();
    sample_probes();
  }

  void clear_probe_history
  ()
  { probe_time.clear();
    probe_proper_time.clear();
    probe_E_theta.clear();
  }

  void sample_probes
  ()
  { probe_time.push_back(time);
    for (double r : probe_radii)
    { probe_proper_time.push_back(Schwarzschild_lapse(M, r)*time);
      probe_E_theta    .push_back(sample_centered_local_E_theta(r));
    }
  }

  [[nodiscard]] double sample_centered_local_E_theta
  ( double r
  ) const
  { return pic_1d::Bspline<0>::gather_from_cells(grid, r,
      [&](size_t i)
      { return maxwell_free_1d::local_E_th(grid.cells[i], fields.cells[i].D_th);
      });
  }

  void advance_until
  ( double t_stop,
    size_t sample_stride
  )
  { const size_t n_steps_f = ceil((t_stop-time)/dt);
    advance(n_steps_f, sample_stride);
  }

  void advance
  ( size_t steps,
    size_t sample_stride
  )
  { py::gil_scoped_release release;
    for (size_t i = 0; i < steps; i++)
    { maxwell_free_1d::step_Maxwell_vac(grid, fields, dt, boundary);
      time += dt;
      step++;
      if (sample_stride && step%sample_stride == 0)
        sample_probes();
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] Diagnostics diagnostics
( const Simulation& sim
)
{ const auto max_abs_component =
  []<typename A, typename F>
  ( const A* x,
    size_t n,
    const F& f
  )
  { double out = 0;
    for (size_t i = 0; i < n; i++)
      out = std::max(out, std::abs(f(x[i])));
    return out;
  };

  return
  { .time = sim.time,
    .step = sim.step,
    .dt = sim.dt,
    .dr = sim.grid.dr,
    .energy_EM =
      maxwell_free_1d::diagnostics::energy_EM(sim.grid, sim.fields),
    .max_abs_D_theta =
      max_abs_component(
        sim.fields.cells.get(), sim.fields.n_cells,
        [](const auto& f) { return f.D_th; }),
    .max_abs_D_phi =
      max_abs_component(
        sim.fields.cells.get(), sim.fields.n_cells,
        [](const auto& f) { return f.D_phi; }),
    .max_abs_B_theta =
      max_abs_component(
        sim.fields.faces.get(), sim.fields.n_faces,
        [](const auto& f) { return f.B_th; }),
    .max_abs_B_phi =
      max_abs_component(
        sim.fields.faces.get(), sim.fields.n_faces,
        [](const auto& f) { return f.B_phi; }),
  };
}

////////////////////////////////////////////////////////////////////////////////

// Passing data to Python.
// Some of this is in-memory, some of it is newly created.

[[nodiscard]] py::object owner
( Simulation& sim
)
{ return py::cast(&sim, py::return_value_policy::reference);
}

[[nodiscard]] py::array_t<double> r_cell
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.cells[0].x.r, sim.grid.n_cells,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> r_face
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.faces[0].x.r, sim.grid.n_faces,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> alpha_cell
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.cells[0].m.geo.alpha, sim.grid.n_cells,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> alpha_face
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.faces[0].m.geo.alpha, sim.grid.n_faces,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> weight_cell
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.cells[0].reduction.w_vol, sim.grid.n_cells,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> weight_face
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.grid.faces[0].reduction.w_vol, sim.grid.n_faces,
    static_cast<py::ssize_t>(sizeof(Grid::MetricPoint)), owner(sim));
}

[[nodiscard]] py::array_t<double> D_theta
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.fields.cells[0].D_th, sim.fields.n_cells,
    static_cast<py::ssize_t>(sizeof(maxwell_free_1d::MaxwellYeeGrid::Cell)),
    owner(sim));
}

[[nodiscard]] py::array_t<double> D_phi
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.fields.cells[0].D_phi, sim.fields.n_cells,
    static_cast<py::ssize_t>(sizeof(maxwell_free_1d::MaxwellYeeGrid::Cell)),
    owner(sim));
}

[[nodiscard]] py::array_t<double> B_theta
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.fields.faces[0].B_th, sim.fields.n_faces,
    static_cast<py::ssize_t>(sizeof(maxwell_free_1d::MaxwellYeeGrid::Face)),
    owner(sim));
}

[[nodiscard]] py::array_t<double> B_phi
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.fields.faces[0].B_phi, sim.fields.n_faces,
    static_cast<py::ssize_t>(sizeof(maxwell_free_1d::MaxwellYeeGrid::Face)),
    owner(sim));
}

[[nodiscard]] py::array_t<double> local_E_theta
( Simulation& sim
)
{ auto* data = new double[sim.grid.n_cells];
  for (size_t i = 0; i < sim.grid.n_cells; i++)
    data[i] =
      maxwell_free_1d::local_E_th(sim.grid.cells[i], sim.fields.cells[i].D_th);
  std::mdspan view {data, sim.grid.n_cells};
  return util::to_py_array(view);
}

[[nodiscard]] py::array_t<double> local_E_phi
( Simulation& sim
)
{ auto* data = new double[sim.grid.n_cells];
  for (size_t i = 0; i < sim.grid.n_cells; i++)
    data[i] =
      maxwell_free_1d::local_E_phi(sim.grid.cells[i], sim.fields.cells[i].D_phi);
  std::mdspan view {data, sim.grid.n_cells};
  return util::to_py_array(view);
}

[[nodiscard]] py::array_t<double> local_B_theta
( Simulation& sim
)
{ auto* data = new double[sim.grid.n_faces];
  for (size_t i = 0; i < sim.grid.n_faces; i++)
    data[i] =
      maxwell_free_1d::local_B_th(sim.grid.faces[i], sim.fields.faces[i].B_th);
  std::mdspan view {data, sim.grid.n_faces};
  return util::to_py_array(view);
}

[[nodiscard]] py::array_t<double> local_B_phi
( Simulation& sim
)
{ auto* data = new double[sim.grid.n_faces];
  for (size_t i = 0; i < sim.grid.n_faces; i++)
    data[i] =
      maxwell_free_1d::local_B_phi(sim.grid.faces[i], sim.fields.faces[i].B_phi);
  std::mdspan view {data, sim.grid.n_faces};
  return util::to_py_array(view);
}

[[nodiscard]] py::array_t<double> probe_times
( Simulation& sim
)
{ return util::to_py_array_ref(
    sim.probe_time.data(), sim.probe_time.size(), owner(sim));
}

[[nodiscard]] py::array_t<double> probe_radii
( Simulation& sim
)
{ return util::to_py_array_ref(
    sim.probe_radii.data(), sim.probe_radii.size(), owner(sim));
}

[[nodiscard]] py::array_t<double> probe_proper_times
( Simulation& sim
)
{ std::mdspan view
  { sim.probe_proper_time.data(),
    sim.probe_time.size(),
    sim.probe_radii.size()
  };
  return util::to_py_array_ref(view, owner(sim));
}

[[nodiscard]] py::array_t<double> probe_local_E_theta
( Simulation& sim
)
{ std::mdspan view
  { sim.probe_E_theta.data(),
    sim.probe_time.size(),
    sim.probe_radii.size()
  };
  return util::to_py_array_ref(view, owner(sim));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(GR_redshift, m)
{

m.def
( "redshift_ratio",
  &redshift_ratio,
  py::arg("M"),
  py::arg("r_emit"),
  py::arg("r_obs")
);

py::class_<Diagnostics>(m, "Diagnostics")
  .def_readonly("time", &Diagnostics::time)
  .def_readonly("step", &Diagnostics::step)
  .def_readonly("dt", &Diagnostics::dt)
  .def_readonly("dr", &Diagnostics::dr)
  .def_readonly("field_energy", &Diagnostics::energy_EM)
  .def_readonly("max_abs_D_theta", &Diagnostics::max_abs_D_theta)
  .def_readonly("max_abs_D_phi", &Diagnostics::max_abs_D_phi)
  .def_readonly("max_abs_B_theta", &Diagnostics::max_abs_B_theta)
  .def_readonly("max_abs_B_phi", &Diagnostics::max_abs_B_phi);

py::class_<Simulation>(m, "Simulation")
  .def
  ( py::init<double, double, double, size_t, double, std::vector<double>>(),
    py::arg("M"),
    py::arg("r_min"),
    py::arg("r_max"),
    py::arg("n_cells"),
    py::arg("cfl"),
    py::arg("probe_radii")
  )
  .def_readwrite("M", &Simulation::M)
  .def_readwrite("dt", &Simulation::dt)
  .def_readwrite("time", &Simulation::time)
  .def_readwrite("step", &Simulation::step)
  .def("initialize",
    &Simulation::initialize,
    py::arg("r_center"),
    py::arg("width"),
    py::arg("omega"))
  .def("advance",
    &Simulation::advance,
    py::arg("steps"),
    py::arg("sample_stride"))
  .def("advance_until",
    &Simulation::advance_until,
    py::arg("t_stop"),
    py::arg("sample_stride"))
  .def("sample_probes", &Simulation::sample_probes)
  .def("clear_probe_history", &Simulation::clear_probe_history)
  .def("diagnostics", &diagnostics)
  .def("r_cell", &r_cell)
  .def("r_face", &r_face)
  .def("alpha_cell", &alpha_cell)
  .def("alpha_face", &alpha_face)
  .def("weight_cell", &weight_cell)
  .def("weight_face", &weight_face)
  .def("D_theta", &D_theta)
  .def("D_phi", &D_phi)
  .def("B_theta", &B_theta)
  .def("B_phi", &B_phi)
  .def("local_E_theta", &local_E_theta)
  .def("local_E_phi", &local_E_phi)
  .def("local_B_theta", &local_B_theta)
  .def("local_B_phi", &local_B_phi)
  .def("probe_times", &probe_times)
  .def("probe_proper_times", &probe_proper_times)
  .def("probe_local_E_theta", &probe_local_E_theta)
  .def("probe_radii", &probe_radii);
}
