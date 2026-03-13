#include <cstddef>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<double>
make_array
( py::ssize_t n
)
{ auto* data = new double[static_cast<size_t>(n)];
  for (py::ssize_t i = 0; i < n; i++)
    data[i] = static_cast<double>(i)/(n-1);

  auto owner = py::capsule(data, [](void* ptr)
  { delete[] static_cast<double*>(ptr);
  });

  return py::array_t<double>(
// shape
    {n},
// strides in bytes
    {static_cast<py::ssize_t>(sizeof(double))},
    data, owner);
}

PYBIND11_MODULE(test_B1, m)
{ m.def("make_array", &make_array, py::arg("n"));
}
