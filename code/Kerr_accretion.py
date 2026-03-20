#!/usr/bin/env python

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

dir_this = Path(__file__).resolve().parent
path_out = dir_this/'out'
sys.path.insert(0, str(path_out))
import Kerr_accretion as kerr_accretion

################################################################################

res = (256, 256)
camera_dist = 30
camera_tilt = np.pi/18
camera_target = (0, 0, 0)
camera_focal_ratio = .7
disk_inclination = np.pi/9
disk_radius = (4.5, 14)

# res = (256, 256)
# camera_dist = 30
# camera_tilt = np.pi/18
# camera_target = (0, 4, 0)
# camera_focal_ratio = .15
# disk_inclination = np.pi/9
# disk_radius = (4.5, 14)

M = 1.
a = .7
eps = 0

dt = .02
max_steps = 20000
domain_L = 60.
h_rel = 1e-5
h_min = 1e-6

################################################################################

colors, stop_criteria, iteration_counts = kerr_accretion.render(
  *res,
  camera_dist, camera_tilt, *camera_target,
  camera_focal_ratio,
  disk_inclination, *disk_radius,
  M, a, eps,
  dt,
  max_steps,
  domain_L,
  h_rel, h_min,
)

stop_criterion = [
  'none',
  'max_steps',
  'invalid_state',
  'horizon_entry',
  'domain_exit',
  'accretion_disk'
]
print('     count stop_reason')
for criterion in np.unique(stop_criteria):
  count = int(np.count_nonzero(stop_criteria == criterion))
  print(f'  {count:>8} {stop_criterion[criterion]:>10}')

################################################################################

plt.imsave(path_out/'Kerr_acc_1.png', colors)

cmap = plt.get_cmap('gray')
norm = plt.Normalize(iteration_counts.min(), iteration_counts.max())
plt.imsave(path_out/'Kerr_acc_2.png', cmap(norm(iteration_counts)))
