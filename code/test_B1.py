#!/usr/bin/env python

'''
Together with `test_B1.cpp`, this shows how to effectively construct a NumPy
array from C++, and pass it (with ownership of the data) to Python.
'''

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'out'))

import test_B1

if __name__ == '__main__':
  n = 8
  x: np.ndarray = test_B1.make_array(n)
  print(f'{x.dtype=}')
  print(f'{x.shape=}')
  print(f'{x.strides=}')
  print(f'{x=}')
