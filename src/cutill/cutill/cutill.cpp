#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "myutill.hpp"
#include "DVODGMRF.hpp"

using namespace std;
namespace py = pybind11;

// 二つの信号を[-1, 1]に正規化
py::tuple normalize(py::array_t<double>& _data1, py::array_t<double>& _data2) {
	// numpy配列をvectorに変換
	vector<double> data1 = ndarray_to_vector(_data1);
	vector<double> data2 = ndarray_to_vector(_data2);

	// 最大値＆最小値の探索
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

	// 正規化
	#pragma omp parallel for
	for (int i = 0; i < data1.size(); i++) {
		data1[i] = (2 * data1[i] - minim - maxim) / (maxim - minim);
		data2[i] = (2 * data2[i] - minim - maxim) / (maxim - minim);
	}

	return py::make_tuple(vector_to_ndarray(data1), vector_to_ndarray(data2));
}

py::tuple gmrf_denoise(py::array_t<double>& _data1, py::array_t<double>& _data2, int scale) {
	// numpy配列をvectorに変換
	vector<double> data1 = ndarray_to_vector(_data1);
	vector<double> data2 = ndarray_to_vector(_data2);

	// 最大値＆最小値の探索
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

	// [0, scale] 正規化
	#pragma omp parallel for
	for (int i = 0; i < data1.size(); i++) {
		data1[i] = scale * (data1[i] - data1_min) / (data1_max - data1_min);
		data2[i] = scale * (data2[i] - data2_min) / (data2_max - data2_min);
	}

	// gmrf denoise
	DVGMRF::odgmrf<double> gmrf1, gmrf2;
	vector<double> denoised_data1 = gmrf1.processBlock({data1});
	vector<double> denoised_data2 = gmrf2.processBlock({data2});

	// [0, scale] で正規化した信号をもとに戻し、[-1,1]正規化
	#pragma omp parallel for
	for (int i = 0; i < data1.size(); i++) {
		denoised_data1[i] = denoised_data1[i]/scale * (data1_max - data1_min) + data1_min;
		denoised_data1[i] = (2 * denoised_data1[i] - minim - maxim) / (maxim - minim);
		denoised_data2[i] = denoised_data2[i]/scale * (data2_max - data2_min) + data2_min;
		denoised_data2[i] = (2 * denoised_data2[i] - minim - maxim) / (maxim - minim);
	}

	return py::make_tuple(vector_to_ndarray(denoised_data1), vector_to_ndarray(denoised_data2));
}

// 配列の要素が全て同じならtrue
bool is_identical_element(py::array_t<double>& _data) {
	// numpy配列をvectorに変換
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
// モジュール化
PYBIND11_MODULE(cutill, m) {
	m.def("normalize", &normalize);
	m.def("is_identical_element", &is_identical_element);
	m.def("gmrf_denoise", &gmrf_denoise);
}