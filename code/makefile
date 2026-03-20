CXX ?= g++
PYTHON ?= python

# Output directory where compiled binaries and plots and such go. This will be
# removed with `make clean`.
OUT_DIR ?= out

# Provide support for using a libstdc++ (GCC) or libc++ (clang) implementation
# that is not the default OS one. Useful in particular if the OS compiler is too
# old to support C++23.
# Example usage:
#   LLVM_DIR="$HOME/.local/bin/LLVM-21.1.3-Linux-X64"
#   CXX="$LLVM_DIR/bin/clang++"
#   LIBCXX_DIR="$LLVM_DIR/lib/x86_64-unknown-linux-gnu"
#   LIBCXX=libc++
FLAGS_CXXSTDLIB :=
ifneq ($(strip $(LIBCXX)),)
FLAGS_CXXSTDLIB := -stdlib=$(LIBCXX)
endif
CXXSTDLIB_LD_FLAGS :=
ifneq ($(strip $(LIBCXX_DIR)),)
CXXSTDLIB_LD_FLAGS := -L$(LIBCXX_DIR) -Wl,-rpath,$(LIBCXX_DIR)
endif

# We use C++23 for our sanity. See that your compiler supports it, it is
# somewhat new so it may not. It should work on WSL by default since it uses
# Ubuntu and the default Ubuntu now should come with a sytem compiler that
# supports C++23.
CXX_FLAGS := -std=c++23 -O3 $(FLAGS_CXXSTDLIB)
PY_MOD_FLAGS := -fPIC -shared

# OpenMP is disabled by default. Enable it with `OMP=1 make ...`.
OMP ?= 0
OMP_FLAGS :=
ifeq ($(OMP),1)
OMP_FLAGS := -fopenmp
endif

# For Python interoperability, we must link against the Python C library. The
# Python install knows where libpython lives, so this is automatic.
# TODO: Linking Python in this way is not good, pybind11 has kind of 'its own
#   way' that it wants to do this, but I had issues with it before and this does
#   work. Need to test this later and test the proper alternative.
INC_PY := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
FLAGS_LD_PY := $(shell $(PYTHON) -c "import sysconfig; sh=bool(sysconfig.get_config_var('PY_ENABLE_SHARED')); libdir=sysconfig.get_config_var('LIBDIR') or ''; ver=sysconfig.get_config_var('VERSION') or ''; print(f'-L{libdir} -lpython{ver}' if sh and libdir and ver else '')")
PY_EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# pybind11 include directory. Either use the specified one or assume it was pip
# installed and do it that way.
INC_PYBIND11 ?=
ifeq ($(strip $(INC_PYBIND11)),)
INC_PYBIND11 := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
endif
FLAGS_PYBIND11 :=
ifneq ($(strip $(INC_PYBIND11)),)
FLAGS_PYBIND11 := -I$(INC_PYBIND11)
endif

all: Kerr_geodesics Kerr_accretion

.PHONY: all clean Kerr_geodesics Kerr_accretion

clean:
	rm -rf $(OUT_DIR)

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

Kerr_geodesics: $(OUT_DIR)/Kerr_geodesics$(PY_EXT)
$(OUT_DIR)/Kerr_geodesics$(PY_EXT): Kerr_geodesics.cpp | $(OUT_DIR)
	$(CXX) $(CXX_FLAGS) $(PY_MOD_FLAGS) -I$(INC_PY) $(FLAGS_PYBIND11) $< -o $@ $(CXXSTDLIB_LD_FLAGS) $(FLAGS_LD_PY)

Kerr_accretion: $(OUT_DIR)/Kerr_accretion$(PY_EXT)
$(OUT_DIR)/Kerr_accretion$(PY_EXT): Kerr_accretion.cpp | $(OUT_DIR)
	$(CXX) $(CXX_FLAGS) $(PY_MOD_FLAGS) $(OMP_FLAGS) -I$(INC_PY) $(FLAGS_PYBIND11) $< -o $@ $(CXXSTDLIB_LD_FLAGS) $(FLAGS_LD_PY)
