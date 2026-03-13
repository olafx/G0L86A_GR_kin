CXX := g++
CXX_FLAGS := -std=c++23 -O3

PYTHON := python
PY_MOD_FLAGS := -fPIC -shared -undefined dynamic_lookup

INC_PYBIND11 := $(HOME)/off/repos/pybind11/include

INC_PY := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
PY_EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

OUT_DIR := out

all: test_B1

.PHONY: all clean test_B1

clean:
	rm -rf out

out:
	@mkdir -p out

test_B1: out/test_B1$(PY_EXT_SUFFIX)

out/test_B1$(PY_EXT_SUFFIX): test_B1.cpp | out
	$(CXX) $(CXX_FLAGS) $(PY_MOD_FLAGS) -I$(INC_PYBIND11) -I$(INC_PY) $< -o $@
