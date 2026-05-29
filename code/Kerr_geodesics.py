#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

dir_this = Path(__file__).resolve().parent
path_out = dir_this/'out'
sys.path.insert(0, str(path_out))
import Kerr_geodesics as kerr_geodesics
import util

################################################################################

M = 1.
a = .7
dt = .02
max_steps = 50_000
domain_L = 20.
h_rel = 1e-5
h_min = 1e-6
method = 'IMR_split'
iters_IMR = 4
assert method in ('RK4', 'IMR', 'IMR_split')

# x_r, x_th, x_phi, u_r, u_th, u_phi, eps
IC = kerr_geodesics.IC
ics = [
  IC( 8.00, 0.50*np.pi, +0.00, +0.00, +0.00, +3.80, 1),
  IC(10.00, 0.38*np.pi, +0.00, -0.03, +0.50, +4.80, 1),
  IC( 6.00, 0.50*np.pi, +0.00, -0.24, +0.00, +2.50, 1),
  IC(12.00, 0.50*np.pi, -1.10, -1.20, +0.00, +7.50, 0),
]
labels = [
  'timelike equatorial',
  'timelike inclined',
  'timelike plunge',
  'lightlike scatter',
]

################################################################################

match method:
  case 'RK4':
    geos, geos_meta = kerr_geodesics.geodesics_RK4(
      M, a,
      dt,
      max_steps,
      domain_L,
      h_rel, h_min,
      ics,
    )
  case 'IMR':
    geos, geos_meta = kerr_geodesics.geodesics_IMR(
      M, a,
      dt,
      max_steps, iters_IMR,
      domain_L,
      h_rel, h_min,
      ics,
    )
  case 'IMR_split':
    geos, geos_meta = kerr_geodesics.geodesics_IMR_split(
      M, a,
      dt,
      max_steps, iters_IMR,
      domain_L,
      h_rel, h_min,
      ics,
    )

stop_criterion = [
  'none',
  'max_steps',
  'invalid_state',
  'horizon_entry',
  'domain_exit',
]
print('i_geo  steps stop')
for i_geo, (geo, geo_meta) in enumerate(zip(geos, geos_meta)):
  print(
    f'{i_geo:>5} {geo_meta.steps:>6} '
    f'{stop_criterion[geo_meta.stop_criterion]}'
  )

################################################################################

# Horizons in Kerr-Schild Cartesian coordinates:
#   (x^2+y^2)/(r_h^2+a^2)+z^2/r_h^2 = 1

alpha_horizon = .2
n_horizon = 64

r_m = M-(M**2-a**2)**.5
r_p = M+(M**2-a**2)**.5
horizons = []
th = np.linspace(0, np.pi, n_horizon//2)
phi = np.linspace(0, 2*np.pi, n_horizon)
for r_h in (r_m, r_p):
  rho_h = (r_h**2+a**2)**.5
  z_h = r_h
  xs = rho_h*np.outer(np.cos(phi), np.sin(th))
  ys = rho_h*np.outer(np.sin(phi), np.sin(th))
  zs = z_h*np.outer(np.ones_like(phi), np.cos(th))
  horizons += [(rho_h, z_h, xs, ys, zs)]

################################################################################

plt.figure(figsize=(5, 5))
for i_geo, path in enumerate(geos):
  path_x = path[:, 0, :]
  path_Car = util.sph_to_Car(*path_x.T)
  plt.plot(path_Car[0], path_Car[1], label=labels[i_geo])
for rho_h, _, _, _, _ in horizons:
  plt.gca().add_patch(
    Ellipse((0, 0), width=2*rho_h, height=2*rho_h,
      facecolor='#000000', edgecolor='none', alpha=alpha_horizon))
plt.xlim(-domain_L, +domain_L)
plt.ylim(-domain_L, +domain_L)
plt.gca().set_aspect(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(path_out/'Kerr_paths_xy.png', dpi=800, bbox_inches='tight')
plt.close()

plt.figure(figsize=(5, 5))
for i_geo, path in enumerate(geos):
  path_x = path[:, 0, :]
  path_Car = util.sph_to_Car(*path_x.T)
  plt.plot(path_Car[0], path_Car[2], label=labels[i_geo])
for rho_h, z_h, _, _, _ in horizons:
  plt.gca().add_patch(
    Ellipse((0, 0), width=2*rho_h, height=2*z_h,
      facecolor="#000000", edgecolor='none', alpha=alpha_horizon))
plt.xlim(-domain_L, +domain_L)
plt.ylim(-domain_L, +domain_L)
plt.gca().set_aspect(1)
plt.xlabel('$x$')
plt.ylabel('$z$')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(path_out/'Kerr_paths_xz.png', dpi=800, bbox_inches='tight')
plt.close()

################################################################################

plt.figure(figsize=(5, 5))
plt.axes(projection='3d')
plt.gca().view_init(30, -70)
for i_geo, path in enumerate(geos):
  path_x = path[:, 0, :]
  path_Car = util.sph_to_Car(*path_x.T)
  plt.plot(*path_Car, label=labels[i_geo])
for _, _, xs, ys, zs in horizons:
  plt.gca().plot_surface(xs, ys, zs, color='#000000',
    alpha=alpha_horizon, shade=False)
plt.xlim(-domain_L, +domain_L)
plt.ylim(-domain_L, +domain_L)
plt.gca().set_zlim(-domain_L, +domain_L)
plt.gca().set_box_aspect((1, 1, 1))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.gca().set_zlabel('$z$')
plt.legend(loc='upper left', fontsize=8)
plt.gca().grid(False)
plt.gca().xaxis.pane.fill = False
plt.gca().yaxis.pane.fill = False
plt.gca().zaxis.pane.fill = False
plt.tight_layout()
plt.savefig(path_out/'Kerr_paths.png', dpi=800, bbox_inches='tight')
plt.close()
