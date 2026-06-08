#include <algorithm>
#include <cmath>
#include <memory>
#include <mdspan>
#include <numbers>
#include <random>

#include "diagnostics_1d.hpp"
#include "util.hpp"
#include "Vlasov_Poisson_1d.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace
{

namespace py = pybind11;
using Grid = maxwell_1d::SchwarzschildBLRadialGrid;
using Fields = maxwell_1d::MaxwellYeeGrid;
using Particle = pic_1d::Particle<pic_1d::Bspline<0>>;

////////////////////////////////////////////////////////////////////////////////

struct Diagnostics
{ double time;
  size_t step;
  double dt;
  size_t n_particles;
  size_t n_mobile;
  double total_charge;
  double max_abs_D_r;
  double max_abs_densitized_rho;
  double energy_EM;
  double energy_kinetic;
};

struct TwoStreamParams
{ double n0;
  double v_beam;
  double v_perturb;
  double v_sd;
  size_t perturb_mode;
  size_t nppc_per_beam;
};

////////////////////////////////////////////////////////////////////////////////

// Local velocity perturbation of a single mode.
[[nodiscard]] double local_vel_mode_delta
( const Grid& grid,
  const TwoStreamParams& params,
  int charge_sign,
  double r
)
{ const double L = grid.r_max-grid.r_min;
  const double phase =
    2*std::numbers::pi*params.perturb_mode*(r-grid.r_min)/L;
  return charge_sign*params.v_perturb*sin(phase);
}

// Two-stream local velocity from some standard Gaussian noise.
[[nodiscard]] double init_local_vel
( const Grid& grid,
  const TwoStreamParams& params,
  int charge_sign,
  int stream_sign,
  double r,
  double rand_Gauss
)
{ const double clean = stream_sign*params.v_beam;
  const double mode = local_vel_mode_delta(grid, params, charge_sign, r);
  const double noise = charge_sign*params.v_sd*rand_Gauss;
  return clean+mode+noise;
}

// Compute the appropriate macroparticle masses for an electron-positron pair.
// 
// NOTE: Divided by two because it's for electron-positron pairs.
// NOTE: There is no electron mass, we take |q|=m=c=1. `params.n0` effectively
//   incorporates some kind of electron mobility, that determines the physics.
[[nodiscard]] double init_macro_pair_mass
( const Grid& grid,
  const TwoStreamParams& params,
  size_t i_cell
)
{ return .5*grid.cells[i_cell].reduction.w_vol*grid.dr
    /params.nppc_per_beam*params.n0;
}

// Initialize a macroparticle pairs, via multiple calls to this.
void init_macro_pair
( const Grid& grid,
  const TwoStreamParams& params,
  Particle& p,
  int charge_sign,
  int stream_sign,
  double r,
  double rand_Gauss = 0
)
{ const size_t i_cell = std::min(
    static_cast<size_t>(floor((r-grid.r_min)/grid.dr)), grid.n_cells-1);
  const double mass = init_macro_pair_mass(grid, params, i_cell);
  const Vec3_sph x {r, std::numbers::pi/2, 0};
  const metric::Kerr::BoyerLindquist metric { .params = grid.metric_params };
  const auto m = metric.template eval<true>(x);
  const double v_hat_r = init_local_vel(
    grid, params, charge_sign, stream_sign, r, rand_Gauss);

  p =
  { .x = x,
    .u = {pic_1d::radial_local_v_to_cov_u(m, v_hat_r), 0, 0},
    .species =
    { .eps = 1,
      .q_over_m = static_cast<double>(charge_sign),
    },
    .r_prev = r,
    .m = mass,
    .charge_sign = charge_sign,
    .label = stream_sign,
    .mobile = true,
  };
}

////////////////////////////////////////////////////////////////////////////////

// Particle boundary conditions, i.e. ways of dealing with particles that move
// outside of the domain.

// Reinsert on the other side as a driven beam.
//
// NOTE: Macroparticle mass is maintained, so charge is maintained.
struct TwoStreamReinjection
{
  TwoStreamParams& params;

  void operator()
  ( const Grid& grid,
    Particle& p
  ) const
  { if (p.x.r >= grid.r_min && p.x.r < grid.r_max)
      return;
    const int charge_sign = p.charge_sign;
    const int stream_sign = p.label;
    const double mass = p.m;
    const auto species = p.species;
    const double L = grid.r_max-grid.r_min;
    pic_1d::wrap_to_domain(grid.r_min, L, p);
    init_macro_pair(grid, params, p, charge_sign, stream_sign, p.x.r);
    p.m = mass;
  }
};

// Absorb them into the boundary and disable them.
//
// NOTE: Macroparticle mass is not maintained, so charge is not maintained.
struct AbsorbingBoundary
{
  void operator()
  ( const Grid& grid,
    Particle& p
  ) const
  { if (p.x.r >= grid.r_min && p.x.r < grid.r_max)
      return;
    p.mobile = false;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct Simulation
{
  double M;
  Grid grid;
  Fields fields;
  size_t n_particles;
  std::unique_ptr<Particle[]> particles;
  std::unique_ptr<double[]> densitized_rho;
  TwoStreamParams two_stream;
  solve::IMR solver;
  finite_difference::policies::Simple policy_fd;
  AbsorbingBoundary boundary;
  double dt;
  double time;
  size_t step;

  Simulation
  ( double M,
    double r_min,
    double r_max,
    size_t n_cells,
    size_t nppc_per_beam,
    double n0,
    double v_beam,
    double v_perturb,
    double v_sd,
    size_t perturb_mode,
    double cfl,
    double tol,
    size_t n_iter,
    double h_rel,
    double h_min
  ) : M(M),
      grid(maxwell_1d::make_grid
        <maxwell_1d::SchwarzschildBLRadialGrid::ThinFluxTubeReduction>(
        metric::Kerr::Params {M, 0}, n_cells, r_min, r_max)),
      fields(
        maxwell_1d::make_fields(grid)),
      n_particles(
        4*n_cells*nppc_per_beam),
      particles(
        std::make_unique_for_overwrite<Particle[]>(n_particles)),
      densitized_rho(
        std::make_unique_for_overwrite<double[]>(n_cells)),
      two_stream
      { .n0 = n0,
        .v_beam = v_beam,
        .v_perturb = v_perturb,
        .v_sd = v_sd,
        .perturb_mode = perturb_mode,
        .nppc_per_beam = nppc_per_beam,
      },
      solver
      { .dt = cfl*grid.dr/maxwell_1d::max_radial_light_speed(grid),
        .tol = tol,
        .n_iter = n_iter,
      },
      policy_fd
      { .h_rel = h_rel,
        .h_min = h_min
      },
      dt(solver.dt),
      time(0),
      step(0)
  { initialize_quiet();
    vlassov_poisson_1d::deposit_charge_and_solve_Gauss(
      grid, fields, particles.get(), n_particles, densitized_rho.get(),
      [](const Particle& p) { return p.mobile; }, 0);
  }

// Quiet initialization, with exactly opposing electrons and positrons.
  void initialize_quiet
  ()
  { std::mt19937 rng {2036-8-12};
    std::normal_distribution<double> normal {0, 1};
    size_t p = 0;
    for (size_t i = 0; i < grid.n_cells; i++)
    for (size_t j = 0; j < two_stream.nppc_per_beam; j++)
    { const double xi = (j+.5)/two_stream.nppc_per_beam;
      const double r = grid.r_min+(i+xi)*grid.dr;
      const double g_R = normal(rng);
      const double g_L = normal(rng);
      init_macro_pair(grid, two_stream, particles[p++], -1, +1, r, g_R);
      init_macro_pair(grid, two_stream, particles[p++], +1, +1, r, g_R);
      init_macro_pair(grid, two_stream, particles[p++], -1, -1, r, g_L);
      init_macro_pair(grid, two_stream, particles[p++], +1, -1, r, g_L);
    }
  }

  void advance_until
  ( double t_stop
  )
  { if (t_stop <= time)
      return;
    const double n_steps_f = ceil((t_stop-time)/dt);
    advance(n_steps_f);
  }

  void advance
  ( size_t steps
  )
  { py::gil_scoped_release release;
    for (size_t i = 0; i < steps; i++)
    { vlassov_poisson_1d::step(
        grid, fields, particles.get(), n_particles,
        densitized_rho.get(),
        solver, policy_fd, boundary);
      time += dt;
      step++;
    }
// One last deposition and Gauss solve before computing diagnostics.
    vlassov_poisson_1d::deposit_charge_and_solve_Gauss(
      grid, fields, particles.get(), n_particles, densitized_rho.get(),
      [](const Particle& p) { return p.mobile; }, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] Diagnostics diagnostics
( const Simulation& sim
)
{ double total_charge = 0;
  double max_abs_densitized_rho = 0;
#pragma omp parallel for schedule(static)\
reduction(+:total_charge) reduction(max:max_abs_densitized_rho)
  for (size_t i = 0; i < sim.grid.n_cells; i++)
  { total_charge += sim.densitized_rho[i]*sim.grid.dr;
    max_abs_densitized_rho =
      std::max(max_abs_densitized_rho, std::abs(sim.densitized_rho[i]));
  }

  double max_abs_D_r = 0;
#pragma omp parallel for schedule(static) reduction(max:max_abs_D_r)
  for (size_t i = 0; i < sim.grid.n_faces; i++)
    max_abs_D_r = std::max(max_abs_D_r, std::abs(sim.fields.faces[i].D_r));

  size_t n_mobile = 0;
#pragma omp parallel for schedule(static) reduction(+:n_mobile)
  for (size_t i = 0; i < sim.n_particles; i++)
    n_mobile += sim.particles[i].mobile;

  const double energy_kinetic = maxwell_1d::diagnostics::energy_kinetic(
    sim.grid, sim.particles.get(), sim.n_particles);
  const double energy_EM      = maxwell_1d::diagnostics::energy_EM(
    sim.grid, sim.fields);

  return
  { .time = sim.time,
    .step = sim.step,
    .dt = sim.dt,
    .n_particles = sim.n_particles,
    .n_mobile = n_mobile,
    .total_charge = total_charge,
    .max_abs_D_r = max_abs_D_r,
    .max_abs_densitized_rho = max_abs_densitized_rho,
    .energy_EM = energy_EM,
    .energy_kinetic = energy_kinetic,
  };
}

////////////////////////////////////////////////////////////////////////////////

// Passing data to Python.

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

[[nodiscard]] py::array_t<double> particle_r
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].x.r, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<double> particle_u_r
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].u.r, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<double> particle_m
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].m, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<double> particle_q_over_m
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].species.q_over_m, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<int> particle_charge_sign
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].charge_sign, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<int> particle_stream_sign
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].label, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<bool> particle_mobile
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.particles[0].mobile, sim.n_particles,
    static_cast<py::ssize_t>(sizeof(Particle)), owner(sim));
}

[[nodiscard]] py::array_t<double> D_r
( Simulation& sim
)
{ return util::to_py_array_ref(
    &sim.fields.faces[0].D_r, sim.fields.n_faces,
    static_cast<py::ssize_t>(sizeof(Fields::Face)), owner(sim));
}

[[nodiscard]] py::array_t<double> densitized_rho
( Simulation& sim
)
{ return util::to_py_array_ref(
    sim.densitized_rho.get(), sim.grid.n_cells, owner(sim));
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(two_stream_Poisson, m)
{

py::class_<Diagnostics>(m, "Diagnostics")
  .def_readonly("time", &Diagnostics::time)
  .def_readonly("step", &Diagnostics::step)
  .def_readonly("dt", &Diagnostics::dt)
  .def_readonly("n_particles", &Diagnostics::n_particles)
  .def_readonly("n_mobile", &Diagnostics::n_mobile)
  .def_readonly("total_charge", &Diagnostics::total_charge)
  .def_readonly("max_abs_D_r", &Diagnostics::max_abs_D_r)
  .def_readonly(
    "max_abs_densitized_rho", &Diagnostics::max_abs_densitized_rho)
  .def_readonly("energy_EM", &Diagnostics::energy_EM)
  .def_readonly("energy_kinetic", &Diagnostics::energy_kinetic);

py::class_<Simulation>(m, "Simulation")
  .def
  ( py::init<
      double, double, double, size_t, size_t,
      double, double, double, double, size_t,
      double, double, size_t, double, double>(),
    py::arg("M"),
    py::arg("r_min"),
    py::arg("r_max"),
    py::arg("n_cells"),
    py::arg("nppc_per_beam"),
    py::arg("n0"),
    py::arg("v_beam"),
    py::arg("v_perturb"),
    py::arg("v_sd"),
    py::arg("perturb_mode"),
    py::arg("cfl"),
    py::arg("tol"),
    py::arg("n_iter"),
    py::arg("h_rel"),
    py::arg("h_min")
  )
  .def_readwrite("dt", &Simulation::dt)
  .def_readwrite("time", &Simulation::time)
  .def_readwrite("step", &Simulation::step)
  .def("advance", &Simulation::advance, py::arg("steps"))
  .def("advance_until", &Simulation::advance_until, py::arg("t_stop"))
  .def("diagnostics", &diagnostics)
  .def("r_cell", &r_cell)
  .def("r_face", &r_face)
  .def("particle_r", &particle_r)
  .def("particle_u_r", &particle_u_r)
  .def("particle_m", &particle_m)
  .def("particle_q_over_m", &particle_q_over_m)
  .def("particle_charge_sign", &particle_charge_sign)
  .def("particle_stream_sign", &particle_stream_sign)
  .def("particle_mobile", &particle_mobile)
  .def("D_r", &D_r)
  .def("densitized_rho", &densitized_rho);
}
