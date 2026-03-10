CXX := g++
CXXFLAGS := -std=c++23 -O3

OUT_DIR := out

all: test_1

.PHONY: all clean test_1

clean:
	rm -rf out

out:
	@mkdir -p out

test_1: out/test_1

out/test_1: test_1.cpp | out
	$(CXX) $(CXXFLAGS) $< -o $@
