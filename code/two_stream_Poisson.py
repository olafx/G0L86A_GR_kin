#!/usr/bin/env python

from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

dir_this = Path(__file__).resolve().parent
dir_out = dir_this/'out'
dir_frames = dir_out/'two_stream_Poisson'
dir_out.mkdir(exist_ok=True)
dir_frames.mkdir(exist_ok=True)
sys.path.insert(0, str(dir_out))
import two_stream_Poisson as tsP

################################################################################

M = 5e-3
r_min = 1e0
r_max = 1e1
n_cells = 2048
nppc_per_beam = 16 # per species per beam
n0 = 1e-2
v_beam = 1.5e-2
v_perturb = 1e-3 # opposite-sign velocity seed for electrons/positrons
v_sd = 1e-3
perturb_mode = 10 # Fourier mode k = 2*pi*perturb_mode/(r_max-r_min)
cfl = 2e-1
tol = 1e-11
n_iter = 12
h_rel = 1e-5
h_min = 1e-7

t_end = 100
n_frames = 200+1
frame_dpi = 400
video_fps = 10
density_bins = 64
v_range = (-.1, +.05)
p_range = (-0.03, +0.3)

################################################################################

sim = tsP.Simulation(
  M, r_min, r_max, n_cells, nppc_per_beam,
  n0, v_beam, v_perturb, v_sd, perturb_mode,
  cfl, tol, n_iter, h_rel, h_min)

def plot_phase_space(i):
  r = np.array(sim.particle_r())
  u = np.array(sim.particle_u_r())
  mass = np.array(sim.particle_m())
  q_over_m = np.array(sim.particle_q_over_m())
  charge_sign = np.array(sim.particle_charge_sign())
  stream_sign = np.array(sim.particle_stream_sign())
  mobile = np.array(sim.particle_mobile())

  fig, ax = plt.subplots(figsize=(7, 4))
  for q_sign, s_sign, color, label in [
    (-1, +1, '#3B82F6', r'$e^-$, $L\to R$ beam'),
    (+1, +1, '#22C55E', r'$e^+$, $L\to R$ beam'),
    (-1, -1, '#F59E0B', r'$e^-$, $R\to L$ beam'),
    (+1, -1, '#EC4899', r'$e^+$, $R\to L$ beam'),
  ]:
    keep = mobile & (charge_sign == q_sign) & (stream_sign == s_sign)
    ax.plot(
      r[keep], u[keep], '.', alpha=.3, ms=1, lw=0,
      color=color, label=label)

  bin_edges = np.linspace(r_min, r_max, density_bins+1)
  bin_centers = .5*(bin_edges[:-1]+bin_edges[1:])
  dr_bin = np.diff(bin_edges)
  r_mobile = r[mobile]
  number_counts, _ = np.histogram(r_mobile, bins=bin_edges)
  if number_counts.sum() > 0:
    number_density = number_counts/(number_counts.sum()*dr_bin)
  else:
    number_density = np.zeros_like(bin_centers)

  charge = mass*q_over_m
  charge_abs_total = np.sum(np.abs(charge[mobile]))
  charge_counts, _ = np.histogram(
    r_mobile, bins=bin_edges, weights=charge[mobile])
  if charge_abs_total > 0:
    charge_density = charge_counts/(charge_abs_total*dr_bin)
  else:
    charge_density = np.zeros_like(bin_centers)

  ax_density = ax.twinx()
  ax_density.step(
    bin_centers, number_density, where='mid',
    c='black', lw=1.0, alpha=.75, label=r'$n$')
  ax_density.step(
    bin_centers, charge_density, where='mid',
    c='tab:purple', lw=1.0, alpha=.75, label=r'$\rho_q$')
  ax_density.set_ylabel(r'normalized density')
  ax_density.set_ylim(*p_range)

  ax.set_title(fr'$t={sim.time:.2f}$')
  ax.set_xlabel(r'$r$')
  ax.set_ylabel(r'$u_r$')
  ax.set_xlim(r_min, r_max)
  ax.set_ylim(*v_range)
  lines, labels = ax.get_legend_handles_labels()
  lines_density, labels_density = ax_density.get_legend_handles_labels()
  ax.legend(
    lines+lines_density, labels+labels_density,
    loc='lower right', markerscale=6)
  fig.tight_layout()
  fig.savefig(dir_frames/f'{i:06d}.png', dpi=frame_dpi)
  plt.close(fig)

t_frames = np.linspace(0, t_end, n_frames)
diag_time = []
diag_energy_kinetic = []
diag_energy_EM = []

for i, t_frame in enumerate(t_frames):
  sim.advance_until(t_frame)
  diag = sim.diagnostics()
  diag_time.append(diag.time)
  diag_energy_kinetic.append(diag.energy_kinetic)
  diag_energy_EM.append(diag.energy_EM)
  plot_phase_space(i)

subprocess.run([
  'ffmpeg',
  '-y',
  '-framerate', str(video_fps),
  '-i', str(dir_frames/'%06d.png'),
  '-frames:v', str(n_frames),
  '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
  '-pix_fmt', 'yuv420p',
  str(dir_out/'two_stream_Poisson.mp4'),
], check=True)

plt.figure(figsize=(6, 3))
plt.plot(diag_time, diag_energy_kinetic, c='black')
plt.xlim(0, t_end)
plt.xlabel(r'$t$'); plt.ylabel(r'$K$')
plt.yscale('log')
plt.tight_layout()
plt.savefig(dir_out/'two_stream_Poisson_K.png', dpi=frame_dpi)

plt.figure(figsize=(6, 3))
plt.plot(diag_time, diag_energy_EM, c='black')
plt.xlim(0, t_end)
plt.xlabel(r'$t$'); plt.ylabel(r'$U_E$')
plt.yscale('log')
plt.tight_layout()
plt.savefig(dir_out/'two_stream_Poisson_U_E.png', dpi=frame_dpi)

diag = sim.diagnostics()
print(f'step={diag.step}')
print(f'time={diag.time:.6g}')
print(f'dt={diag.dt:.6g}')
print(f'n_particles={diag.n_particles}')
print(f'n_mobile={diag.n_mobile}')
print(f'total_charge={diag.total_charge:.4e}')
print(f'max_abs_D_r={diag.max_abs_D_r:.4e}')
print(f'max_abs_densitized_rho={diag.max_abs_densitized_rho:.4e}')
print(f'energy_kinetic={diag.energy_kinetic:.4e}')
print(f'energy_EM={diag.energy_EM:.4e}')
