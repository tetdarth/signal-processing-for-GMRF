#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace std;

// 1次元numpy配列をvectorに変換
std::vector<double> ndarray_to_vector(py::array_t<double>& array) {
    // バッファ情報を取得
    py::buffer_info buf_info = array.request();
    double* ptr = static_cast<double*>(buf_info.ptr);

    // std::vectorに変換
    std::vector<double> vec(ptr, ptr + buf_info.size);

    return vec;
}

// 1次元vectorをnumpy配列に変換
py::array_t<double> vector_to_ndarray(vector<double>& vec) {
    // vectorのサイズを取得
    size_t size = vec.size();
    py::array_t<double> array(size);
    py::buffer_info buf_info = array.request();

    // データポインタを取得
    double* ptr = static_cast<double*>(buf_info.ptr);
    // データコピー
    std::memcpy(ptr, vec.data(), size * sizeof(double));

    return array;
}


// 2次元numpy配列をvectorに変換
std::vector<std::vector<double>> ndarray_to_vector_2d(py::array_t<double>& x)
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

    return result;
}

// 2次元vectorをnumpy配列に変換
py::array_t<double> vector_to_ndarray_2d(const std::vector<double>& vec) {
    py::array_t<double> arr(vec.size());
    auto buffer = arr.request();
    double* ptr = static_cast<double*>(buffer.ptr);
    for (size_t i = 0; i < vec.size(); ++i) ptr[i] = vec[i];

    return arr;
}
