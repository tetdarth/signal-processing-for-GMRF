#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "include.hpp"

namespace py = pybind11;

/* numpy”z—ñ‚©‚çstd::vector‚É•ÏŠ· */

class ndarray_vec
{
protected:
    // ndarray -> vector
    std::vector<std::vector<double>> ndarray_to_vector(py::array_t<double> x)
    {
        const auto& buff_info = x.request();
        const auto& shape = buff_info.shape;

        std::vector<std::vector<double>> result(shape[0], std::vector<double>(shape[1]));

        for (auto i = 0; i < shape[0]; i++) {
            for (auto j = 0; j < shape[1]; j++) {
                auto v = *x.data(i, j);
                result[i][j] = static_cast<double>(v);
            }
        }
        /*
        for (const auto& v : result) {
            for (const auto& p : v) {
                cout << p << ",";
            }
            cout << endl;
        }
        */
        return result;
    }

    // vector -> ndarray
    py::array_t<double> vector_to_ndarray(const std::vector<double>& vec) {
        py::array_t<double> arr(vec.size());
        auto buffer = arr.request();
        double* ptr = static_cast<double*>(buffer.ptr);
        for (size_t i = 0; i < vec.size(); ++i) ptr[i] = vec[i];

        return arr;
    }
};
