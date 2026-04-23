#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <bit>
#include <cmath>
#include <vector>



#include "util.hpp"
#include "solve.hpp"
#include "geodesic.hpp"
#include "metric.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace
{

namespace py = pybind11;

// There is some casting shenanigans due to `solve.hpp` working with general
// sized vectors, while the state is represented as a pair of 3-vectors, i.e.
// a 2x3 matrix Mat23.
using Vec3 = util::Vec<double, 3>;
using Vec6 = util::Vec<double, 6>;
using Mat23 = util::Ten<double, 3, 2>::V;

// The initial condition conceptually is an object at rest in some reference
// frame and at some event. The object's identity is sufficiently characterized
// by whether it has mass or not (`eps`). Its rest reference frame is
// characterized by a velocity (boost) since the object is pointlike. The event
// is characterized only by a position since the metric is time-independent.
struct IC
{ Mat23 state;
  double eps;
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
( const metric::Kerr::Params& params_Kerr,
  double domain_L,
  const Vec3& x
)
{ for (auto e : x.data)
    if (!std::isfinite(e))
      return StopCriterion::invalid_state;
  if (x.r <= params_Kerr.r_horizon.p)
    return StopCriterion::horizon_entry;
  else if (is_outside_domain(domain_L, x))
    return StopCriterion::domain_exit;
  else [[likely]]
    return StopCriterion::none;
}

////////////////////////////////////////////////////////////////////////////////

// TODO: Supporting different ODE integrators, in particular the symplectic
// implicit midpoint rule with Picard iteration is next. Need to look into what
// the best way is to handle that, because the parameters will be slightly
// different, but also don't want to copy paste all the code, and the switching
// should also be at compile time, like there should be no slow check in the
// main loop.
py::tuple geodesics
( const std::vector<IC>& ics,
  double M,
  double a,
  double dt,
  int max_steps,
  double domain_L,
  double h_rel,
  double h_min,
  int n_iter
)
{ const metric::Kerr::Params params_Kerr {M, a};
  const metric::Kerr::BoyerLindquist metric;
  const finite_difference::StepPolicy_Simple step_policy {h_rel, h_min};

// `geodesics` is a Python list that will contain geodesic arrays.
// `geodesics_meta` is the corresponding metadata of the geodesics.
  py::list geodesics(ics.size());
  py::list geodesics_meta(ics.size());

  {
// Release the GIL for the computational work, don't need it.
    py::gil_scoped_release release;

// Iterate over each geodesic initial condition.
    #pragma omp parallel for
    for (size_t i_geodesic = 0; i_geodesic < ics.size(); i_geodesic++)
    { const IC& ic = ics[i_geodesic];
// Allocate the geodesic array through spacetime. The time coordinate is
// implicit since the integrator has a fixed time step.
      std::vector<Vec3> geodesic;
      geodesic.reserve(max_steps+1);
      geodesic.push_back(ic.state.X);

// Prepare the ODE integrator.
// The RHS function for the ODE integrator works on vectors (Vec6), while the
// geodesic RHS describes a position and velocity evolution (Mat23), so we must
// do some casting.
      Mat23 state = ic.state;
      StopCriterion stop_criterion = StopCriterion::max_steps;
      auto rhs = [&](const Vec6& y)
      { return std::bit_cast<Vec6>(
          geodesic::rhs(
            params_Kerr, metric, ic.eps, std::bit_cast<Mat23>(y), step_policy));
      };

// The ODE integrator main loop. Check for the stop event.
      size_t i_step = 0;
      for (; i_step < max_steps; i_step++)
      { const Vec3& x = state.X;
        stop_criterion = stop_event(params_Kerr, domain_L, x);
        if (stop_criterion != StopCriterion::none)
          break;
        state = std::bit_cast<Mat23>(
          n_iter == 0
          ? solve::rk4_step(rhs, dt, std::bit_cast<Vec6>(state))
          : solve::implicit_midpoint_step(rhs, dt, std::bit_cast<Vec6>(state), n_iter));
        geodesic.push_back(state.X);
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
        geodesics[i_geodesic] = util::to_py_array(std::move(geodesic));
        geodesics_meta[i_geodesic] = GeodesicMeta
        { .steps = i_step,
          .stop_criterion = std::to_underlying(stop_criterion)
        };
      }
    }
  }
  return py::make_tuple(geodesics, geodesics_meta);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(Kerr_geodesics, m)
{

// The IC and GeodesicMeta classes need to be defined in Python also.

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
          double eps
        )
        { return IC
          { .state =
            { {x_r, x_th, x_phi},
              {u_r, u_th, u_phi}
            },
            .eps = eps,
          };
        }
      ),
      py::arg("x_r"),
      py::arg("x_th"),
      py::arg("x_phi"),
      py::arg("u_r"),
      py::arg("u_th"),
      py::arg("u_phi"),
      py::arg("eps")
    );

  py::class_<GeodesicMeta>(m, "GeodesicMeta")
    .def_readonly("steps", &GeodesicMeta::steps)
    .def_readonly("stop_criterion", &GeodesicMeta::stop_criterion);

  m.def
  ( "geodesics", &geodesics,
    py::arg("ics"),
    py::arg("M"),
    py::arg("a"),
    py::arg("dt"),
    py::arg("max_steps"),
    py::arg("domain_L"),
    py::arg("h_rel"),
    py::arg("h_min"),
    py::arg("n_iter") = 0
  );
}
