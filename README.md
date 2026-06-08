# G0L86A_GR_kin

## Structure

`code` contains the code, `report` contains the report. Both have a `makefile`
for convenience.

All outputs (compiled programs and images and such) will end up in `code/out`,
to keep the directory structure clean. `make clean` will delete this entire
directory so take care to remove output data from there.

## Dependencies

- pybind11: To more easily return NumPy arrays and such from C++. `pip install
pybind11` should work with no further setup, or you can clone the repository and
set `INC_PYBIND11` to the include folder.
- OpenMP: For parallelization. It is optional but is very useful. `libomp` is
not needed so it should already come with the compiler. We use `-fopenmp`, which
should work on clang and GCC.
- Python C library: Needed to bind C to Python. But this should be dealt with 
automatically. Just make sure the `python` on the path is the one you intend to
run with, i.e. see that your virtual environment is activated, it will figure
out the rest automatically.

## Language server

`.vscode/settings.json` is set up so that the language server will work if you
use `compiledb` or `bear` to produce a `compile_commands.json` file. So instead
of `make`, use `compiledb make` or `bear -- make`, and it will produce this
`compile_commands.json` file, and IntelliSense will work in VS Code. The
`makefile` is too convoluted for VS Code to decrypt the meaning of on its own,
so it will not work with the VS Code makefile plugin otherwise.

## How to run?

### `Kerr_geodesics`

```bash
cd code
make Kerr_geodesics
./Kerr_geodesics.py
```

### `Kerr_accretion`

Best ran with OpenMP parallelization for performance, but it is optional.
```bash
cd code
OMP=1 make Kerr_accretion
./Kerr_accretion.py # on all threads
OMP_NUM_THREADS=42 ./Kerr_accretion.py # on 42
make Kerr_accretion
./Kerr_accretion.py # on 1 thread
```

### `GR_redshift`

Vacuum 1D-3V Maxwell redshift test on a radial Schwarzschild grid.
```bash
cd code
OMP=1 make GR_redshift
./GR_redshift.py # on all threads
```

### `two_stream_Poisson`

Electron-positron two-stream instability on a radial Schwarzschild grid, for the
Vlasov-Poisson model.
```bash
cd code
OMP=1 make two_stream_Poisson
./two_stream_Poisson.py # on all threads
```
