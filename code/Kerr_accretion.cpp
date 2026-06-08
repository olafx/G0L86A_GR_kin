#include "util.hpp"
#include "common.hpp"
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
  const size_t2& i,
  const size_t2& res
)
{ const double sx = ((i.x+.5)/res.x-.5)*camera.focal_ratio*res.x/res.y;
  const double sy = ((i.y+.5)/res.y-.5)*camera.focal_ratio;
  return normalized(camera.forward+sx*camera.right-sy*camera.up);
}

// Consider a point x and a direction u^i for some observer at x. x is described
// in spherical coordinates, u^i in Cartesian coordinates. Convert this to u_i
// in spherical coordinates.
// First we compute the spherical basis vectors, then project against them
// via the spatial metric gamma.
//
// NOTE: This u_i in spherical coordinates still needs to be converted to
//   Boyer-Lindquist coordinates in principle, but for simplicity we don't do
//   this. It is a good approximation at large distances, since Boyer-Lindquist
//   reduces appropriately to spherical coordinates.
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
//
// NOTE: This uses the flat spacetime metric for r to compare against the disk
//   radius r. This is not correct (really depending on what we even mean by the
//   disk radius), but it is a good approximation if the inner disk radius is
//   far away from the event horizon. (Really the disk should be defined by
//   theta=pi/2 and r in some range, in Boyer-Lindquist coordinates. That is the
//   correct generalization of a flat circular disk in Euclidian space. And what
//   the generalization of that is with inclination, I have no idea.)
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
( const metric::Kerr::Params& metric_params,
  const AccretionDisk& disk,
  double domain_L,
  const Vec3_sph& x_prev,
  const Vec3_sph& x  
)
{ for (auto e : x.data)
    if (!std::isfinite(e)) [[unlikely]]
      return StopCriterion::invalid_state;
  if (x.r <= metric_params.r_horizon.p) [[unlikely]]
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
    return {0x00, 0x00, 0x00}; // black
}

////////////////////////////////////////////////////////////////////////////////

template
< typename Metric,
  typename Particle,
  typename EMField,
  typename Scheme
>
py::tuple render
( const geodesic::Problem<Metric, Particle, EMField>& problem,
  const AccretionDisk& disk,
  const Scheme& scheme,
  size_t max_steps,
  double domain_L,
  const Camera& camera,
  size_t2 res
)
{
// We return the main colored image, an image of stop criteria, and an image of
// the number of iterations.
  auto* buf_img           = new RGB[product(res)];
  auto* buf_stop_criteria = new int[product(res)];
  auto* buf_iterations    = new int[product(res)];
  std::mdspan img              {buf_img,           res.y, res.x};
  std::mdspan stop_criteria    {buf_stop_criteria, res.y, res.x};
  std::mdspan iteration_counts {buf_iterations,    res.y, res.x};

  {
// Release the GIL for the computational work.
    py::gil_scoped_release release;

    const Vec3_sph camera_x = util::Car_to_sph(camera.x);

#pragma omp parallel for schedule(dynamic)
    for (size_t i_y = 0; i_y < res.y; i_y++)
    for (size_t i_x = 0; i_x < res.x; i_x++)
    {
// Preparation for the ODE integrator.
      Mat23 state =
      { camera_x,
        direction_to_u_cov(
          camera_x, pixel_direction(camera, {i_x, i_y}, res))
      };
      scheme.initialize(problem, state);
      Vec3_sph x_prev = state.X;
      StopCriterion stop_criterion = StopCriterion::max_steps;

// The ODE integrator main loop. Check for the stop event.
      size_t i_step = 0;
      for (; i_step < max_steps; i_step++)
      { const Vec3_sph& x = state.X;
        stop_criterion = stop_event(
          problem.metric.params, disk, domain_L, x_prev, x);
        if (stop_criterion != StopCriterion::none)
          break;
        x_prev = x;
        scheme.step(problem, state);
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

////////////////////////////////////////////////////////////////////////////////

py::tuple render_RK4
( double metric_M,
  double metric_a,
  double disk_inclination,
  double disk_r_inner,
  double disk_r_outer,
  double eps,
  double h_rel,
  double h_min,
  double dt,
  size_t max_steps,
  double domain_L,
  double camera_dist,
  double camera_tilt,
  double camera_target_x,
  double camera_target_y,
  double camera_target_z,
  double camera_focal_ratio,
  size_t res_x,
  size_t res_y
)
{ const metric::Kerr::BoyerLindquist metric
  { .params = {metric_M, metric_a}
  };
  constexpr em::Vacuum em_field {};
  const geodesic::particle::Neutral p {eps};
  const AccretionDisk disk
  { .normal = {0, sin(disk_inclination),
                  cos(disk_inclination)},
    .r = {disk_r_inner,
          disk_r_outer}
  };
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  const solve::RK4 stepper {dt};
  const Camera camera = make_camera
  ( camera_dist, camera_tilt, camera_focal_ratio,
    { camera_target_x,
      camera_target_y,
      camera_target_z
    }
  );
  const size_t2 res {res_x, res_y};
  const geodesic::schemes::Full scheme {stepper};
  const geodesic::Problem problem {metric, em_field, policy_fd, p};
  return render(problem, disk, scheme, max_steps, domain_L, camera, res);
}

py::tuple render_IMR
( double metric_M,
  double metric_a,
  double disk_inclination,
  double disk_r_inner,
  double disk_r_outer,
  double eps,
  double h_rel,
  double h_min,
  double dt,
  size_t max_steps,
  size_t IMR_max_iters,
  double domain_L,
  double camera_dist,
  double camera_tilt,
  double camera_target_x,
  double camera_target_y,
  double camera_target_z,
  double camera_focal_ratio,
  size_t res_x,
  size_t res_y
)
{ const metric::Kerr::BoyerLindquist metric
  { .params = {metric_M, metric_a}
  };
  constexpr em::Vacuum em_field {};
  const geodesic::particle::Neutral p {eps};
  const AccretionDisk disk
  { .normal = {0, sin(disk_inclination),
                  cos(disk_inclination)},
    .r = {disk_r_inner,
          disk_r_outer}
  };
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const Camera camera = make_camera
  ( camera_dist, camera_tilt, camera_focal_ratio,
    { camera_target_x,
      camera_target_y,
      camera_target_z
    }
  );
  const size_t2 res {res_x, res_y};
  const geodesic::schemes::Full scheme {stepper};
  const geodesic::Problem problem {metric, em_field, policy_fd, p};
  return render(problem, disk, scheme, max_steps, domain_L, camera, res);
}

py::tuple render_IMR_split
( double metric_M,
  double metric_a,
  double disk_inclination,
  double disk_r_inner,
  double disk_r_outer,
  double eps,
  double h_rel,
  double h_min,
  double dt,
  size_t max_steps,
  size_t IMR_max_iters,
  double domain_L,
  double camera_dist,
  double camera_tilt,
  double camera_target_x,
  double camera_target_y,
  double camera_target_z,
  double camera_focal_ratio,
  size_t res_x,
  size_t res_y
)
{ const metric::Kerr::BoyerLindquist metric
  { .params = {metric_M, metric_a}
  };
  constexpr em::Vacuum em_field {};
  const geodesic::particle::Neutral p {eps};
  const AccretionDisk disk
  { .normal = {0, sin(disk_inclination),
                  cos(disk_inclination)},
    .r = {disk_r_inner,
          disk_r_outer}
  };
  const finite_difference::policies::Simple policy_fd {h_rel, h_min};
  constexpr double IMR_tol = 1e-12;
  const solve::IMR stepper {dt, IMR_tol, IMR_max_iters};
  const Camera camera = make_camera
  ( camera_dist, camera_tilt, camera_focal_ratio,
    { camera_target_x,
      camera_target_y,
      camera_target_z
    }
  );
  const size_t2 res {res_x, res_y};
  const geodesic::schemes::Split scheme {stepper};
  const geodesic::Problem problem {metric, em_field, policy_fd, p};
  return render(problem, disk, scheme, max_steps, domain_L, camera, res);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(Kerr_accretion, m)
{

m.def(
  "render_RK4",
  &render_RK4,
  py::arg("metric_M"),
  py::arg("metric_a"),
  py::arg("disk_inclination"),
  py::arg("disk_r_inner"),
  py::arg("disk_r_outer"),
  py::arg("eps"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("domain_L"),
  py::arg("camera_dist"),
  py::arg("camera_tilt"),
  py::arg("camera_target_x"),
  py::arg("camera_target_y"),
  py::arg("camera_target_z"),
  py::arg("camera_focal_ratio"),
  py::arg("res_x"),
  py::arg("res_y")
);

m.def(
  "render_IMR",
  &render_IMR,
  py::arg("metric_M"),
  py::arg("metric_a"),
  py::arg("disk_inclination"),
  py::arg("disk_r_inner"),
  py::arg("disk_r_outer"),
  py::arg("eps"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("iters_IMR"),
  py::arg("domain_L"),
  py::arg("camera_dist"),
  py::arg("camera_tilt"),
  py::arg("camera_target_x"),
  py::arg("camera_target_y"),
  py::arg("camera_target_z"),
  py::arg("camera_focal_ratio"),
  py::arg("res_x"),
  py::arg("res_y")
);

m.def(
  "render_IMR_split",
  &render_IMR_split,
  py::arg("metric_M"),
  py::arg("metric_a"),
  py::arg("disk_inclination"),
  py::arg("disk_r_inner"),
  py::arg("disk_r_outer"),
  py::arg("eps"),
  py::arg("h_rel"),
  py::arg("h_min"),
  py::arg("dt"),
  py::arg("max_steps"),
  py::arg("iters_IMR"),
  py::arg("domain_L"),
  py::arg("camera_dist"),
  py::arg("camera_tilt"),
  py::arg("camera_target_x"),
  py::arg("camera_target_y"),
  py::arg("camera_target_z"),
  py::arg("camera_focal_ratio"),
  py::arg("res_x"),
  py::arg("res_y")
);

}
