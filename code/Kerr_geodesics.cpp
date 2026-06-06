#include "util.hpp"
#include "common.hpp"
#include "solve.hpp"
#include "geodesic.hpp"
#include "metric.hpp"
#include "electromagnetic.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace
{

namespace py = pybind11;

// The initial condition conceptually is an object at rest in some reference
// frame and at some event. The object's identity is sufficiently characterized
// by whether it has mass or not (`eps`). Its rest reference frame is
// characterized by a velocity (boost) since the object is pointlike. The event
// is characterized only by a position since the metric is time-independent. And
// it has a ratio q/m relevant for the Lorentz force.
struct IC
{ Mat23 state;
  geodesic::particle::Charged p;
};

// Numerical geodesic metadata.
// `steps` is the number of integration steps, not the length of the output
// array.
// `stop_criterion` is the underlying integer of the `StopCriterion` enum.
struct GeodesicMeta
{ size_t steps;
  int stop_criterion;
};

// Integration of the geodesic is stopped for various reasons.
enum class StopCriterion : int
{ none,
  max_steps,
  invalid_state,
  horizon_entry,
  domain_exit,
};

////////////////////////////////////////////////////////////////////////////////

// Consider the geodesic's spatial domain to be ((-L, +L), (-L, +L), (-L, +L))
// in Kerr-Schild Cartesian coordinates.
[[nodiscard]] bool is_outside_domain
( double L,
  const Vec3& x // Boyer-Lindquist
)
{
// Early exit for performance, the conversion to Cartesian is usually not
// necessary.
  if (x.r < L) [[likely]]
    return false;
  Vec3 x_Car;
  util::sph_to_Car(x, x_Car);
  for (size_t i = 0; i < 3; i++)
    if (x_Car[i] < -L || x_Car[i] > +L) [[unlikely]]
      return true;
  return false;
}

// Stop conditions triggered by an event along the geodesic.
[[nodiscard]] StopCriterion stop_event
( const metric::Kerr::Params& metric_params,
  double domain_L,
  const Vec3& x
)
{ for (auto e : x.data)
    if (!std::isfinite(e))
      return StopCriterion::invalid_state;
  if (x.r <= metric_params.r_horizon.p)
    return StopCriterion::horizon_entry;
  else if (is_outside_domain(domain_L, x))
    return StopCriterion::domain_exit;
  else [[likely]]
    return StopCriterion::none;
}

////////////////////////////////////////////////////////////////////////////////

template <geodesic::particle::Kind Particle, typename Scheme, typename EMField>
py::tuple geodesics
( const metric::Kerr::BoyerLindquist& metric,
  const EMField& em_field,
  const finite_difference::policies::Simple& policy_fd,
  const Scheme& scheme,
  size_t max_steps,
  double domain_L,
  const std::vector<IC>& ics
)
{ 
// `geodesics` is a Python list that will contain geodesic arrays.
// `geodesics_meta` is the corresponding metadata of the geodesics.
  py::list geodesics(ics.size());
  py::list geodesics_meta(ics.size());

  {
// Release the GIL for the computational work.
    py::gil_scoped_release release;

// Iterate over each geodesic initial condition.
    #pragma omp parallel for schedule(dynamic)
    for (size_t i_geo = 0; i_geo < ics.size(); i_geo++)
    { const IC& ic = ics[i_geo];
// Allocate the geodesic array through phase space. The time coordinate is
// implicit since the integrator has a fixed time step.
      std::vector<Mat23> geodesic;
      geodesic.reserve(max_steps+1);
      geodesic.push_back(ic.state);

// Prepare the ODE integrator.
      Particle p {.eps = ic.p.eps};
      if constexpr (geodesic::particle::ChargedKind<Particle>)
        p.q_over_m = ic.p.q_over_m;
      const geodesic::Problem problem {metric, em_field, policy_fd, p};
      Mat23 state = ic.state;
      scheme.initialize(problem, state);
      Mat23 state_prev = state;
      StopCriterion stop_criterion = StopCriterion::max_steps;

// The ODE integrator main loop. Check for the stop event.
      size_t i_step = 0;
      for (; i_step < max_steps; i_step++)
      { stop_criterion = stop_event(metric.params, domain_L, state.X);
        if (stop_criterion != StopCriterion::none)
          break;
        state_prev = state;
        scheme.step(problem, state);
        geodesic.push_back(scheme.output(problem, state_prev, state));
      }
      if (i_step == max_steps)
        stop_criterion = StopCriterion::max_steps;
      else
// Reallocate `geodesic` in case `max_steps` is too large of a buffer.
        geodesic.shrink_to_fit();

// Acquire the GIL for atomic Python list writes.
// Pass the geodesic and geodesic_meta arrays to Python, which becomes e.g. a
// NumPy array.
      { py::gil_scoped_acquire acquire;
        geodesics[i_geo] = util::to_py_array(std::move(geodesic));
        geodesics_meta[i_geo] = GeodesicMeta
        { .steps = i_step,
          .stop_criterion = std::to_underlying(stop_criterion)
        };
      }
    }
  }
  return py::make_tuple(geodesics, geodesics_meta);
}

////////////////////////////////////////////////////////////////////////////////

py::tuple geodesics_RK4_vacuum
( double M, double a,
  double dt,
  size_t max_steps,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Vacuum em_field {};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  const solve::RK4 stepper {dt};
  const geodesic::schemes::Full scheme {stepper};
  return geodesics<geodesic::particle::Neutral>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

py::tuple geodesics_RK4_Wald
( double M, double a,
  double dt,
  size_t max_steps,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics,
  double Wald_B_z, double Wald_B_x, double Wald_Q
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Wald<metric::Kerr::BoyerLindquist> em_field
    {{Wald_B_z, Wald_B_x, Wald_Q, h_rel, h_min}, metric.params};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  const solve::RK4 stepper {dt};
  const geodesic::schemes::Full scheme {stepper};
  return geodesics<geodesic::particle::Charged>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

py::tuple geodesics_IMR_vacuum
( double M, double a,
  double dt,
  size_t max_steps, size_t IMR_max_iters,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Vacuum em_field {};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const geodesic::schemes::Full scheme {stepper};
  return geodesics<geodesic::particle::Neutral>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

py::tuple geodesics_IMR_Wald
( double M, double a,
  double dt,
  size_t max_steps, size_t IMR_max_iters,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics,
  double Wald_B_z, double Wald_B_x, double Wald_Q_charge
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Wald<metric::Kerr::BoyerLindquist> em_field
    {{Wald_B_z, Wald_B_x, Wald_Q_charge, h_rel, h_min}, metric.params};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const geodesic::schemes::Full scheme {stepper};
  return geodesics<geodesic::particle::Charged>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

py::tuple geodesics_IMR_split_vacuum
( double M, double a,
  double dt,
  size_t max_steps, size_t IMR_max_iters,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Vacuum em_field {};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const geodesic::schemes::Split scheme {stepper};
  return geodesics<geodesic::particle::Neutral>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

py::tuple geodesics_IMR_split_Wald
( double M, double a,
  double dt,
  size_t max_steps, size_t IMR_max_iters,
  double domain_L,
  double h_rel, double h_min,
  const std::vector<IC>& ics,
  double Wald_B_z, double Wald_B_x, double Wald_Q
)
{ const metric::Kerr::BoyerLindquist metric {.params = {M, a}};
  const em::Wald<metric::Kerr::BoyerLindquist> em_field
    {{Wald_B_z, Wald_B_x, Wald_Q, h_rel, h_min}, metric.params};
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const geodesic::schemes::Split scheme {stepper};
  return geodesics<geodesic::particle::Charged>(
    metric, em_field, policy_fd, scheme, max_steps, domain_L, ics);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(Kerr_geodesics, m)
{

py::class_<IC>(m, "IC")
  .def
  ( py::init
    ( []
      ( double x_r,
        double x_th,
        double x_phi,
        double u_r,
        double u_th,
        double u_phi,
        double eps,
        double q_over_m
      )
      { return IC
        { .state =
          { {x_r, x_th, x_phi},
            {u_r, u_th, u_phi}
          },
          .p = {eps, q_over_m},
        };
      }
    ),
    py::arg("x_r"),
    py::arg("x_th"),
    py::arg("x_phi"),
    py::arg("u_r"),
    py::arg("u_th"),
    py::arg("u_phi"),
    py::arg("eps"),
    py::arg("q_over_m")
  );

py::class_<GeodesicMeta>(m, "GeodesicMeta")
  .def_readonly("steps", &GeodesicMeta::steps)
  .def_readonly("stop_criterion", &GeodesicMeta::stop_criterion);

m.def
( "geodesics_RK4_vacuum", &geodesics_RK4_vacuum,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics")
);

m.def
( "geodesics_RK4_Wald", &geodesics_RK4_Wald,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics"),
  py::arg("Wald_B_z"),
  py::arg("Wald_B_x"),
  py::arg("Wald_Q")
);

m.def
( "geodesics_IMR_vacuum", &geodesics_IMR_vacuum,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("IMR_max_iters"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics")
);

m.def
( "geodesics_IMR_Wald", &geodesics_IMR_Wald,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("IMR_max_iters"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics"),
  py::arg("Wald_B_z"),
  py::arg("Wald_B_x"),
  py::arg("Wald_Q")
);

m.def
( "geodesics_IMR_split_vacuum", &geodesics_IMR_split_vacuum,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("IMR_max_iters"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics")
);

m.def
( "geodesics_IMR_split_Wald", &geodesics_IMR_split_Wald,
  py::arg("M"),
  py::arg("a"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("IMR_max_iters"),
  py::arg("domain_L"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("ics"),
  py::arg("Wald_B_z"),
  py::arg("Wald_B_x"),
  py::arg("Wald_Q")
);

}
