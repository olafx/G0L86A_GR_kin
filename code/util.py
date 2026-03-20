import numpy as np

def Car_to_sph(x, y, z):
  r = (x**2+y**2+z**2)**.5
  th = np.arccos(z/r)
  phi = np.arctan2(y, x)
  return r, th, phi

def sph_to_Car(r, th, phi):
  x = r*np.sin(th)*np.cos(phi)
  y = r*np.sin(th)*np.sin(phi)
  z = r*np.cos(th)
  return x, y, z
