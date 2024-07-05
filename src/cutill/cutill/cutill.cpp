#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include "myutill.h"

using namespace std;
namespace py = pybind11;

// 二つの信号を[-1, 1]に正規化
py::tuple normalize(py::array_t<double>& _data1, py::array_t<double>& _data2) {
	// numpy配列をvectorに変換
	vector<double> data1 = ndarray_to_vector(_data1);
	vector<double> data2 = ndarray_to_vector(_data2);

	double data1_max = DBL_MIN;
	double data1_min = DBL_MAX;
	double data2_max = DBL_MIN;
	double data2_min = DBL_MAX;
	for (size_t i = 0; i < data1.size(); i++) {
		if (data1_max < data1[i]) data1_max = data1[i];
		if (data1_min > data1[i]) data1_min = data1[i];
		if (data2_max < data2[i]) data2_max = data2[i];
		if (data2_min > data2[i]) data2_min = data2[i];
	}
	double maxim = std::max(data1_max, data2_max);
	double minim = std::min(data1_min, data2_min);
	double gap = maxim - minim;

	for (size_t i = 0; i < data1.size(); i++) {
		data1[i] = (data1[i] - minim) / gap * 2 - 1;
		data2[i] = (data2[i] - minim) / gap * 2 - 1;
	}

	return py::make_tuple(vector_to_ndarray(data1), vector_to_ndarray(data2));
}

// 配列の要素が全て同じならtrue
bool is_identical_element(py::array_t<double>& _data) {
	// numpy配列をvectorに変換
	vector<double> data = ndarray_to_vector(_data);

	double sample = data[1];
	for (int i = 0; i < data.size(); i++) {
		if (sample != data[i]) return false;
	}
	return true;
}

// ==============================================================================
// モジュール化
PYBIND11_MODULE(cutill, m) {
	m.def("normalize", &normalize);
	m.def("is_identical_element", &is_identical_element);
}