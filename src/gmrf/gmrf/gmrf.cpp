#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "IVODGMRF.hpp"
#include "DVODGMRF.hpp"
#include "IVHODGMRF.hpp"
#include "DVHODGMRF.hpp"
#include "include.hpp"

using namespace std;
namespace py = pybind11;

// GMRF module
PYBIND11_MODULE(gmrf, m) {
    pybind11::module module1 = m.def_submodule("ivgmrf");
    pybind11::module module2 = m.def_submodule("dvgmrf");
    pybind11::module module3 = m.def_submodule("ivhgmrf");
    pybind11::module module4 = m.def_submodule("dvhgmrf");

    module1.doc() = "Denoising for Identical Variaces - Gaussian Markov Random Field";
    py::class_<GMRF::ivgmrf_od<double>>(module1, "ivgmrf")
        .def(pybind11::init<>())
        .def_property("_lambda", &GMRF::ivgmrf_od<double>::getLambda, &GMRF::ivgmrf_od<double>::setLambda)
        .def_property("_alpha", &GMRF::ivgmrf_od<double>::getAlpha, &GMRF::ivgmrf_od<double>::setAlpha)
        .def_property("_sigma2", &GMRF::ivgmrf_od<double>::getSigma2, &GMRF::ivgmrf_od<double>::setSigma2)
        .def_property("_epoch", &GMRF::ivgmrf_od<double>::getEpoch, &GMRF::ivgmrf_od<double>::setMaxEpoch)
        .def_property("_lambda_rate", &GMRF::ivgmrf_od<double>::getLambdaRate, &GMRF::ivgmrf_od<double>::setLambdaRate)
        .def_property("_alpha_rate", &GMRF::ivgmrf_od<double>::getAlphaRate, &GMRF::ivgmrf_od<double>::setAlphaRate)
        .def("set_eps", &GMRF::ivgmrf_od<double>::setEps)
        .def("denoise", &GMRF::ivgmrf_od<double>::processBlock)
        ;

    module2.doc() = "Denoising for Different Variaces - Gaussian Markov Random Field";
    py::class_<GMRF::dvgmrf_od<double>>(module2, "dvgmrf")
        .def(pybind11::init<>())
        .def_property("_lambda", &GMRF::dvgmrf_od<double>::getLambda, &GMRF::dvgmrf_od<double>::setLambda)
        .def_property("_alpha", &GMRF::dvgmrf_od<double>::getAlpha, &GMRF::dvgmrf_od<double>::setAlpha)
        .def_property("_epoch", &GMRF::dvgmrf_od<double>::getEpoch, &GMRF::dvgmrf_od<double>::setMaxEpoch)
        .def_property("_sigma2", &GMRF::dvgmrf_od<double>::get_vec_sigma2, &GMRF::dvgmrf_od<double>::setSigma2)
        .def_property("_lambda_rate", &GMRF::dvgmrf_od<double>::getLambdaRate, &GMRF::dvgmrf_od<double>::setLambdaRate)
        .def_property("_alpha_rate", &GMRF::dvgmrf_od<double>::getAlphaRate, &GMRF::dvgmrf_od<double>::setAlphaRate)
        .def("set_eps", &GMRF::dvgmrf_od<double>::setEps)
        .def("denoise", &GMRF::dvgmrf_od<double>::processBlock)
        ;

    module3.doc() = "Denoising for Identical Variaces - Gaussian Markov Random Field";
    py::class_<HGMRF::ivhgmrf_od<double>>(module3, "ivhgmrf")
        .def(pybind11::init<>())
        .def_property("_lambda", &HGMRF::ivhgmrf_od<double>::get_lambda, &HGMRF::ivhgmrf_od<double>::set_lambda)
        .def_property("_alpha", &HGMRF::ivhgmrf_od<double>::get_alpha, &HGMRF::ivhgmrf_od<double>::set_alpha)
        .def_property("_gamma2", &HGMRF::ivhgmrf_od<double>::get_gamma2, &HGMRF::ivhgmrf_od<double>::set_gamma2)
        .def_property("_epoch", &HGMRF::ivhgmrf_od<double>::get_epoch, &HGMRF::ivhgmrf_od<double>::set_epoch)
        .def_property("_sigma2", &HGMRF::ivhgmrf_od<double>::get_sigma2, &HGMRF::ivhgmrf_od<double>::set_sigma2)
        .def_property("_lambda_rate", &HGMRF::ivhgmrf_od<double>::get_lambda_rate, &HGMRF::ivhgmrf_od<double>::set_lambda_rate)
        .def_property("_alpha_rate", &HGMRF::ivhgmrf_od<double>::get_alpha_rate, &HGMRF::ivhgmrf_od<double>::set_alpha_rate)
        .def_property("_gamma2_rate", &HGMRF::ivhgmrf_od<double>::get_gamma2_rate, &HGMRF::ivhgmrf_od<double>::set_gamma2_rate)
        .def("set_eps", &HGMRF::ivhgmrf_od<double>::set_eps)
        .def("denoise", &HGMRF::ivhgmrf_od<double>::denoising)
        ;

    module4.doc() = "Denoising for Different Variaces - Gaussian Markov Random Field";
    py::class_<HGMRF::dvhgmrf_od<double>>(module4, "dvhgmrf")
        .def(pybind11::init<>())
        .def_property("_lambda", &HGMRF::dvhgmrf_od<double>::get_lambda, &HGMRF::dvhgmrf_od<double>::set_lambda)
        .def_property("_alpha", &HGMRF::dvhgmrf_od<double>::get_alpha, &HGMRF::dvhgmrf_od<double>::set_alpha)
        .def_property("_gamma2", &HGMRF::dvhgmrf_od<double>::get_gamma2, &HGMRF::dvhgmrf_od<double>::set_gamma2)
        .def_property("_epoch", &HGMRF::dvhgmrf_od<double>::get_epoch, &HGMRF::dvhgmrf_od<double>::set_epoch)
        .def_property("_sigma2", &HGMRF::dvhgmrf_od<double>::get_vec_sigma2, &HGMRF::dvhgmrf_od<double>::set_sigma2)
        .def_property("_lambda_rate", &HGMRF::dvhgmrf_od<double>::get_lambda_rate, &HGMRF::dvhgmrf_od<double>::set_lambda_rate)
        .def_property("_alpha_rate", &HGMRF::dvhgmrf_od<double>::get_alpha_rate, &HGMRF::dvhgmrf_od<double>::set_alpha_rate)
        .def_property("_gamma2_rate", &HGMRF::dvhgmrf_od<double>::get_gamma2_rate, &HGMRF::dvhgmrf_od<double>::set_gamma2_rate)
        .def("set_eps", &HGMRF::dvhgmrf_od<double>::set_eps)
        .def("denoise", &HGMRF::dvhgmrf_od<double>::denoising)
        ;
}