#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "myutill.h"

using namespace std;
namespace py = pybind11;

// ��̐M����[-1, 1]�ɐ��K��
py::tuple normalize(py::array_t<double>& _data1, py::array_t<double>& _data2) {
	// numpy�z���vector�ɕϊ�
	vector<double> data1 = ndarray_to_vector(_data1);
	vector<double> data2 = ndarray_to_vector(_data2);

	// �ő�l���ŏ��l�̒T��
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

	// ���K��
	for (int i = 0; i < data1.size(); i++) {
		data1[i] = (2 * data1[i] - minim - maxim) / (maxim - minim);
		data2[i] = (2 * data2[i] - minim - maxim) / (maxim - minim);
	}

	return py::make_tuple(vector_to_ndarray(data1), vector_to_ndarray(data2));
}

// �z��̗v�f���S�ē����Ȃ�true
bool is_identical_element(py::array_t<double>& _data) {
	// numpy�z���vector�ɕϊ�
	vector<double> data = ndarray_to_vector(_data);

	double sample = data[1];
	bool ans = true;
	#pragma omp parallel for
	for (int i = 0; i < data.size(); ++i) {
		if (sample != data[i]) {
			ans = false;
			break;
		}
	}
	return ans;
}

// ==============================================================================
// ���W���[����
PYBIND11_MODULE(cutill, m) {
	m.def("normalize", &normalize);
	m.def("is_identical_element", &is_identical_element);
}