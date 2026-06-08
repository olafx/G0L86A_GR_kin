#!/usr/bin/env python

from pathlib import Path
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

dir_this = Path(__file__).resolve().parent
dir_out = dir_this/'out'
dir_out.mkdir(exist_ok=True)
sys.path.insert(0, str(dir_out))
import GR_redshift as redshift

################################################################################

M = 1
r_min = 2.05*M # must be >2M
r_max = 105
n_cells = 8192
cfl = 1e-1
assert r_min > 2*M

r_packet = 3
packet_width = 3.2
omega_coordinate = 4

probe_radii = [3.5, 12, 70]
dt_pause = 5
sample_stride = 4
t_end = 92

alpha = .7

################################################################################

def lapse(r):
  return np.sqrt(1-2*M/r)

# NOTE: This is very much AI-generated and I haven't looked at how it works
#   much because it's not interesting for us, as long as it works.
def fitted_omega(tau, y, expected):
  y = np.asarray(y)
  tau = np.asarray(tau)
  if y.size < 8 or np.max(np.abs(y)) == 0:
    return np.nan
  keep = np.abs(y) > 0.12*np.max(np.abs(y))
  if np.count_nonzero(keep) >= 16:
    y = y[keep]
    tau = tau[keep]
  y = y-np.mean(y)
  tau = tau-np.mean(tau)

  def residual(omega):
    A = np.column_stack([
      np.cos(omega*tau),
      np.sin(omega*tau),
      np.ones_like(tau),
    ])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    if not np.all(np.isfinite(coeffs)):
      return np.inf
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
      err = y-A@coeffs
    if not np.all(np.isfinite(err)):
      return np.inf
    return np.mean(err**2)

  omegas = np.linspace(.75*expected, 1.25*expected, 1200)
  i = int(np.argmin([residual(omega) for omega in omegas]))
  lo = omegas[max(0, i-1)]
  hi = omegas[min(len(omegas)-1, i+1)]
  omegas = np.linspace(lo, hi, 200)
  return omegas[int(np.argmin([residual(o) for o in omegas]))]

################################################################################

sim = redshift.Simulation(M, r_min, r_max, n_cells, cfl, probe_radii)
sim.initialize(r_packet, packet_width, omega_coordinate)

while sim.time < t_end:
  sim.advance_until(min(t_end, sim.time+dt_pause), sample_stride)

diag = sim.diagnostics()
print(f'step={diag.step} time={diag.time:.2e} dt={diag.dt:.2e}')
print(f'field_energy={diag.field_energy:.2e}')

probe_tau   = sim.probe_proper_times()
probe_E     = sim.probe_local_E_theta()
probe_radii = sim.probe_radii()

omegas = [fitted_omega(
  probe_tau[:,i], probe_E[:,i], omega_coordinate/lapse(probe_radii[i]))
  for i in range(len(probe_radii))]
omegas = np.array(omegas)

print('probe blueshift vs first probe')
for i, r in enumerate(probe_radii):
  measured = omegas[i]/omegas[0]
  expected = redshift.redshift_ratio(M, probe_radii[0], r)
  print(f'  r={r:6.2e}: measured {measured:.6f} expected {expected:.6f}')

peak_E = np.max(np.abs(probe_E), axis=0)
print('probe peak local E vs first probe')
for i, r in enumerate(probe_radii):
  print(f'  r={r:6.2e}: {peak_E[i]/peak_E[0]:.6f}')

################################################################################

r = sim.r_cell()
E = sim.local_E_theta()

plt.figure(figsize=(6, 3))
plt.plot(r, E, alpha=alpha, c='black')
plt.xlim(r_min, r_max)
plt.xlabel(r'$r$')
plt.ylabel(r'$E^\theta$')
plt.tight_layout()
plt.savefig(dir_out/'GR_redshift_snapshot.png', dpi=400)

plt.figure(figsize=(6, 3))
for i, r_probe in enumerate(probe_radii):
  plt.plot(probe_tau[:,i], probe_E[:,i], label=fr'$r={r_probe:g}$', alpha=alpha)
plt.xlim(np.min(probe_tau), np.max(probe_tau))
plt.xlabel(r'$\tau$')
plt.ylabel(r'$E^\theta$')
plt.legend()
plt.tight_layout()
plt.savefig(dir_out/'GR_redshift_probes.png', dpi=400)

plt.figure(figsize=(6, 4))
for i, r_probe in enumerate(probe_radii):
  i_peak = int(np.argmax(np.abs(probe_E[:,i])))
  tau_peak = probe_tau[i_peak,i]
  tau = probe_tau[:,i]-tau_peak
# filter
  keep = np.abs(tau) < 8
  measured = omegas[i]/omegas[0]
  predicted = redshift.redshift_ratio(M, probe_radii[0], r_probe)
  label = (
    fr'$r={r_probe:g}$, '
    fr'$(\omega/\omega_0)_\mathrm{{fit}}={measured:.3f}$, '
    fr'$(\omega/\omega_0)_\mathrm{{GR}}={predicted:.3f}$'
  )
  plt.plot(tau[keep], probe_E[keep,i]/peak_E[i], label=label, alpha=alpha)
plt.xlim(-8, 8)
plt.ylim(-1.7, 1.1)
plt.xlabel(r'proper time from peak $\Delta\tau$')
plt.ylabel(r'normalized $E^\theta$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(dir_out/'GR_redshift_probe_omega.png', dpi=400)
 