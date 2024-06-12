#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "IVODGMRF.hpp"
#include "IVHODGMRF.hpp"
#include "DVODGMRF.hpp"
#include "DVHODGMRF.hpp"
#include "include.hpp"

using namespace std;
namespace py = pybind11;


PYBIND11_MODULE(gmrf, m) {
    m.doc() = "Denoising for identical variaces - gaussian markov random field";
    py::class_<GMRF::ivgmrf_od<double>>(m, "ivgmrf")
        .def(pybind11::init<>())
        .def_property("lambda", &GMRF::ivgmrf_od<double>::getLambda, &GMRF::ivgmrf_od<double>::setLambda)
        .def_property("alpha", &GMRF::ivgmrf_od<double>::getAlpha, &GMRF::ivgmrf_od<double>::setAlpha)
        .def_property("sigma2", &GMRF::ivgmrf_od<double>::getSigma2, &GMRF::ivgmrf_od<double>::setSigma2)
        .def_property("epoch", &GMRF::ivgmrf_od<double>::getEpoch, &GMRF::ivgmrf_od<double>::setMaxEpoch)
        .def_property("lambda_rate", &GMRF::ivgmrf_od<double>::getLambdaRate, &GMRF::ivgmrf_od<double>::setLambdaRate)
        .def_property("alpha_rate", &GMRF::ivgmrf_od<double>::getAlphaRate, &GMRF::ivgmrf_od<double>::setAlphaRate)
        .def("set_eps", &GMRF::ivgmrf_od<double>::setEps)
        .def("denoise", &GMRF::ivgmrf_od<double>::processBlock)
        ;
}
