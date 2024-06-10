#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector—p
#include <pybind11/operators.h>//operator
#include "pybindtest.hpp"


using namespace::std;

int add(int x, int y) {
    return x + y;
}

namespace py = pybind11;
PYBIND11_MODULE(pybindtest, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add);
    py::class_<POINT>(m, "POINT")
        .def(py::init<int, int>())
        .def(py::init<pair<int, int>>())
        .def_readwrite("sum", &POINT::sum)
        .def_property_readonly("x", &POINT::X)
        .def_property_readonly("y", &POINT::Y)
        .def(py::self + py::self)
        .def("__repr__", &POINT::toString);
}