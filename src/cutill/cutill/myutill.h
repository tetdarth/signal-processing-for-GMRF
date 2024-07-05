#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace std;

// 1����numpy�z���vector�ɕϊ�
std::vector<double> ndarray_to_vector(py::array_t<double>& array) {
    // �o�b�t�@�����擾
    py::buffer_info buf_info = array.request();
    double* ptr = static_cast<double*>(buf_info.ptr);

    // std::vector�ɕϊ�
    std::vector<double> vec(ptr, ptr + buf_info.size);

    return vec;
}

// 1����vector��numpy�z��ɕϊ�
py::array_t<double> vector_to_ndarray(vector<double>& vec) {
    // vector�̃T�C�Y���擾
    size_t size = vec.size();
    py::array_t<double> array(size);
    py::buffer_info buf_info = array.request();

    // �f�[�^�|�C���^���擾
    double* ptr = static_cast<double*>(buf_info.ptr);
    // �f�[�^�R�s�[
    std::memcpy(ptr, vec.data(), size * sizeof(double));

    return array;
}


// 2����numpy�z���vector�ɕϊ�
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

// 2����vector��numpy�z��ɕϊ�
py::array_t<double> vector_to_ndarray_2d(const std::vector<double>& vec) {
    py::array_t<double> arr(vec.size());
    auto buffer = arr.request();
    double* ptr = static_cast<double*>(buffer.ptr);
    for (size_t i = 0; i < vec.size(); ++i) ptr[i] = vec[i];

    return arr;
}
