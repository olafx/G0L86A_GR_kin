#include <bit>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "util.hpp"
#include "solve.hpp"
#include "geodesic.hpp"
#include "metric.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace
{

namespace py = pybind11;

// Let us be explicit about the coordinate system, writing Vec3_sph for
// spherical or Boyer-Lindquist coordinates, and Vec3_Car for Cartesian or
// Kerr-Schild Cartesian coordinates.
using Vec3     = util::Vec<double, 3>;
using Vec6     = util::Vec<double, 6>;
using Mat23    = util::Ten<double, 3, 2>::V;
using Vec3_sph = Vec3;
using Vec3_Car = Vec3;
using Vec2     = util::Vec<double, 2>;
using RGB      = util::Vec<uint8_t, 3>;
using Mat3     = util::Ten<double, 3, 3>::V;

// A camera has a position, an orientation, and some optical/sensor properties.
// It is convenient to specify the orientation by 3 orthonormal vectors.
struct Camera
{ Vec3_Car x;
  Vec3_Car forward, right, up;
  double focal_ratio;
};

// Integration of the geodesic is stopped for various reasons.
enum class StopCriterion : int
{ none,
  max_steps,
  invalid_state,
  horizon_entry,
  domain_exit,
  accretion_disk,
};

// An accretion disk is specified by a normal vector and a set of inner and
// outer radii.
struct AccretionDisk
{ Vec3_Car normal;
  Vec2 r;
};

////////////////////////////////////////////////////////////////////////////////

// Translate the Cartesian coordinate system to the target position. Then the
// camera is placed on the y=0 plane at some distance from the target, and at
// some angle with the x-axis.
[[nodiscard]] Camera make_camera
( double dist,
  double tilt,
  double focal_ratio,
  const Vec3_Car& target
)
{ const Vec3_Car forward {-cos(tilt), 0, -sin(tilt)};
  const Vec3_Car position = target-dist*forward;
  constexpr Vec3_Car right = {0, 1, 0};
  const Vec3_Car up = normalized(cross(right, forward));
  return {position, forward, right, up, focal_ratio};
}

// Spawn geodesics on the 'camera sensor'. Note the signs; the pixels along x in
// an image increase towards +right, the pixels along y towards down=-up.
[[nodiscard]] Vec3 pixel_direction
( const Camera& camera,
  int i_x, int i_y,
  int n_x, int n_y
)
{ const double sx = ((i_x+.5)/n_x-.5)*camera.focal_ratio*n_x/n_y;
  const double sy = ((i_y+.5)/n_y-.5)*camera.focal_ratio;
  return normalized(camera.forward+sx*camera.right-sy*camera.up);
}

// Consider a point x and a direction u^i. x is described in spherical
// coordinates, u^i in Cartesian coordinates. Convert this to u_i in
// spherical coordinates.
// First we compute the spherical basis vectors, then project against them
// via the spatial metric gamma.
[[nodiscard]] Vec3_sph direction_to_u_cov
( const Vec3_sph& x,
  const Vec3_Car& dir
)
{ const Mat3 e = util::e_sph(x);
  return
  { dot(dir, e.r),
    dot(dir, e.th) *x.r,
    dot(dir, e.phi)*x.r*sin(x.th),
  };
}

////////////////////////////////////////////////////////////////////////////////

// Consider the geodesic's spatial domain to be ((-L, +L), (-L, +L), (-L, +L))
// in Kerr-Schild Cartesian coordinates.
[[nodiscard]] bool is_outside_domain
( double L,
  const Vec3_Car& x
)
{ for (size_t i = 0; i < 3; i++)
    if (x[i] < -L || x[i] > +L) [[unlikely]]
      return true;
  return false;
}

// Compute whether the line segment x0x1 intersects with the disk.
//
// Intersection disk with normal n is given by points x such that n*x=0, where
// additionally x lies within the two radii defining the disk.
// Consider 2 points x0, x1, then the segment connecting them is parameterized
// by x0+t(x1-x0) for t in [0, 1]. Considering this instead of an infinite
// line, solve n*(x0+t(x1-x0)) = 0 for t, so t = -n*x0/(n*(x1-x0)). If t is in
// [0, 1], it intersects. Then check if the radius of the resulting intersecting
// point falls in the right range, and be careful of situations where x1-x0 lies
// on the disk.
[[nodiscard]] bool intersects_accretion_disk
( const Vec3_Car& x0,
  const Vec3_Car& x1,
  const AccretionDisk& disk
)
{ constexpr double eps = 1e-12;
  const Vec2 disk_r2 =
  { pow(disk.r.m, 2),
    pow(disk.r.p, 2)
  };
  const Vec3_Car x01 = x1-x0;
// Heights above disk.
  const double  s0 = dot(disk.normal, x0);
  const double  s1 = dot(disk.normal, x1);
  const double s01 = dot(disk.normal, x01);
// x01 is parallel to the disk.
  if (abs(s01) < eps) [[unlikely]]
  {
// See if x0 and x1 are close to the extended disk in n_y.
    if (abs(s0) >= eps ||
        abs(s1) >= eps) [[likely]]
      return false;
// See if their radial distance is right. This means that either one must lie
// on the disk, or they must lie at opposite ends.
    const double r0_2 = norm2(x0);
    const double r1_2 = norm2(x1);
    if      (r0_2 >= disk_r2.m && r0_2 <= disk_r2.p)
      return true;
    else if (r1_2 >= disk_r2.m && r1_2 <= disk_r2.p)
      return true;
    else if (r0_2 <= disk_r2.m && r1_2 >= disk_r2.p)
      return true;
    else if (r1_2 <= disk_r2.m && r0_2 >= disk_r2.p)
      return true;
    else [[likely]]
      return false;
  }
// x01 is not parallel to the disk.
  else
  {
// Consider the param t, see if it lies in the expected range for an
// intersection.
    const double t = -s0/s01;
    if (t < 0 || t > 1) [[likely]]
      return false;
    const Vec3_Car x = x0+t*x01;
// See if the intersection point's radial distance is right.
    const double r = norm2(x);
    return r >= disk_r2.m && r <= disk_r2.p;
  }
}

// Stop criteria triggered by an event along the geodesic.
[[nodiscard]] StopCriterion stop_event
( const metric::Kerr::Params& params_Kerr,
  const AccretionDisk& disk,
  double domain_L,
  const Vec3_sph& x_prev,
  const Vec3_sph& x  
)
{ for (auto e : x.data)
    if (!std::isfinite(e)) [[unlikely]]
      return StopCriterion::invalid_state;
  if (x.r <= params_Kerr.r_horizon.p) [[unlikely]]
    return StopCriterion::horizon_entry;
  const Vec3_Car x_Car = util::sph_to_Car(x);
  if (is_outside_domain(domain_L, x_Car)) [[unlikely]]
    return StopCriterion::domain_exit;
  const Vec3_Car x_prev_Car = util::sph_to_Car(x_prev);
  if (intersects_accretion_disk(x_prev_Car, x_Car, disk)) [[unlikely]]
    return StopCriterion::accretion_disk;
  else
    return StopCriterion::none;
}

////////////////////////////////////////////////////////////////////////////////

// Assign a color to a geodesic based on the camera orientation and stop
// criterion. Moving with the camera is a background of a window of 4 different
// colors, otherwise we see the accretion disk in white and other stop criteria
// in black. We use colors that only vary in hue.
[[nodiscard]] RGB stop_criterion_color
( const Camera& camera,
  const Vec3& x,
  StopCriterion stop_criterion
)
{ if (stop_criterion == StopCriterion::accretion_disk)
    return {0xff, 0xff, 0xff}; // white
  else if (stop_criterion == StopCriterion::domain_exit)
  {
// Project the vector from the camera position to the target onto the left and
// right camera vectors. We only care about the sign of the projections to pick
// the quadrant.
    const Vec3 view = normalized(x-camera.x);
    const double cam_x = dot(view, camera.right);
    const double cam_y = dot(view, camera.up);

    if (cam_x <  0 && cam_y >= 0) // upper-left
      return {0x00, 0x40, 0xff};  // blue
    if (cam_x >= 0 && cam_y >= 0) // upper-right
      return {0xff, 0x00, 0xbf};  // magenta
    if (cam_x <  0 && cam_y <  0) // lower-left
      return {0xff, 0xbf, 0x00};  // orange
    else                          // lower-right
      return {0x00, 0xff, 0x40};  // green
  }
  else
    return {0, 0, 0}; // black
}

////////////////////////////////////////////////////////////////////////////////

// TODO: Supporting different ODE integrators, in particular the symplectic
// implicit midpoint rule with Picard iteration is next. Need to look into what
// the best way is to handle that, because the parameters will be slightly
// different, but also don't want to copy paste all the code, and the switching
// should also be at compile time, like there should be no slow check in the
// main loop.
py::tuple render
( int n_x,
  int n_y,
  double camera_dist,
  double camera_tilt,
  double camera_target_x,
  double camera_target_y,
  double camera_target_z,
  double camera_focal_ratio,
  double disk_inclination,
  double disk_r_inner,
  double disk_r_outer,
  double M,
  double a,
  double eps,
  double dt,
  int max_steps,
  double domain_L,
  double h_rel,
  double h_min
)
{ const metric::Kerr::Params params_Kerr {M, a};
  const metric::Kerr::BoyerLindquist metric;
  const finite_difference::StepPolicy_Simple step_policy {h_rel, h_min};
  const AccretionDisk disk
  { .normal = {0, sin(disk_inclination),
                  cos(disk_inclination)},
    .r = {disk_r_inner,
          disk_r_outer}
  };
  const Camera camera = make_camera
  ( camera_dist, camera_tilt, camera_focal_ratio,
    { camera_target_x,
      camera_target_y,
      camera_target_z
    }
  );
  const Vec3_sph camera_x = util::Car_to_sph(camera.x);

// We return the main colored image, an image of stop criteria, and an image of
// the number of iterations.
  auto* buf_img           = new RGB[n_x*n_y];
  auto* buf_stop_criteria = new int[n_x*n_y];
  auto* buf_iterations    = new int[n_x*n_y];
  std::mdspan img              {buf_img,           n_y, n_x};
  std::mdspan stop_criteria    {buf_stop_criteria, n_y, n_x};
  std::mdspan iteration_counts {buf_iterations,    n_y, n_x};

// Prepare the ODE integrator.
// The RHS function for the ODE integrator works on vectors (Vec6), while the
// geodesic RHS describes a position and velocity evolution (Mat23), so we must
// do some casting.
  auto rhs = [&](const Vec6& y)
  { return std::bit_cast<Vec6>(
      geodesic::rhs(
        params_Kerr, metric, eps, std::bit_cast<Mat23>(y), step_policy));
  };

  {
// Release the GIL for the computational work, don't need it.
    py::gil_scoped_release release;

    #pragma omp parallel for
    for (int i_y = 0; i_y < n_y; i_y++)
      for (int i_x = 0; i_x < n_x; i_x++)
      {
// More preparation for the ODE integrator.
        Mat23 state =
        { camera_x,
          direction_to_u_cov(
            camera_x, pixel_direction(camera, i_x, i_y, n_x, n_y))
        };
        Vec3_sph x_prev = state.X;
        StopCriterion stop_criterion = StopCriterion::max_steps;

// The ODE integrator main loop. Check for the stop event.
        size_t i_step = 0;
        for (; i_step < max_steps; i_step++)
        { const Vec3_sph& x = state.X;
          stop_criterion = stop_event(params_Kerr, disk, domain_L, x_prev, x);
          if (stop_criterion != StopCriterion::none)
            break;
          x_prev = x;
          state = std::bit_cast<Mat23>(
            solve::rk4_step(rhs, dt, std::bit_cast<Vec6>(state)));
        }
        if (i_step == max_steps)
          stop_criterion = StopCriterion::max_steps;

        img[i_y, i_x] = stop_criterion_color(
          camera, util::sph_to_Car(state.X), stop_criterion);
        stop_criteria[i_y, i_x] = std::to_underlying(stop_criterion);
        iteration_counts[i_y, i_x] = i_step;
      }
  }

  return py::make_tuple(
    util::to_py_array(img),
    util::to_py_array(stop_criteria),
    util::to_py_array(iteration_counts)
  );
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(Kerr_accretion, m)
{
  m.def(
    "render",
    &render,
    py::arg("n_x"),
    py::arg("n_y"),
    py::arg("camera_dist"),
    py::arg("camera_tilt"),
    py::arg("camera_target_x"),
    py::arg("camera_target_y"),
    py::arg("camera_target_z"),
    py::arg("camera_focal_ratio"),
    py::arg("disk_inclination"),
    py::arg("disk_r_inner"),
    py::arg("disk_r_outer"),
    py::arg("M"),
    py::arg("a"),
    py::arg("eps"),
    py::arg("dt"),
    py::arg("max_steps"),
    py::arg("domain_L"),
    py::arg("h_rel"),
    py::arg("h_min")
  );
}
