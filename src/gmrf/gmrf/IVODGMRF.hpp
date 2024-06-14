#pragma once

#include "include.hpp"
#include "ndarray_vec.hpp"

#define GSITER 2
// #define DEBUG_MODE

// 1�����f�[�^(One-Dimentional data)�ɑ΂���SVGMRF
namespace GMRF
{
	template<typename Type>
	class ivgmrf_od : public ndarray_vec
	{
		typedef std::vector<Type> vec;
		typedef std::vector<std::vector<Type>> matrix;

	public:
		// =================================================================
		ivgmrf_od(	const Type& _lambda = 1e-11,
					const Type& _alpha = 1e-8,
					const Type& _sigma2 = 5e-01,
					const int& _maxepoch = 1000,
					const Type& _eps = 1e-9,
					const Type& _lambdarate = 1e-12,
					const Type& _alpharate = 1e-6	
		) 
			:	lambda(_lambda),
				alpha(_alpha),
				sigma2(_sigma2),
				maxepoch(_maxepoch),
				eps(_eps),
				lambdaRate(_lambdarate),
				alphaRate(_alpharate)
		{}

		~ivgmrf_od() {}

		// =================================================================
		void setData(const matrix& _data)
		{
			this->enumerate = static_cast<int>(_data.size());
			this->dataSize = static_cast<int>(_data.at(0).size());
			this->eigenvalue = calcEigenVal();
			this->data = centerize2D(_data);
		}

		void gs(const vec& noise, vec& mean)
		{
			const Type inv_sigma2 = enumerate / sigma2;

			// �K�E�X�E�U�C�f���@
			for (uint16_t iter = 0; iter < GSITER; iter++) {
				error = 0.0;

				for (uint16_t i = 0; i < dataSize; i++) {
					// ����
					Type denominator = inv_sigma2 + lambda;
					// ���q
					Type numerator = inv_sigma2 * noise[i];

					// ������̂�GMRF
					if (i + 1 < dataSize) {
						denominator += alpha;
						numerator += alpha * mean[i + 1];
					}
					if (i - 1 >= 0) {
						denominator += alpha;
						numerator += alpha * mean[i - 1];
					}

					const Type currentVal = numerator / denominator;
					error += std::abs(mean[i] - currentVal);
					mean[i] = currentVal;
				}
				error /= dataSize;
			}
		}

		void predParam(const matrix& noise, vec& mean)
		{
			const Type inv_sigma2 = enumerate / sigma2;

			Type lambdaGrad = 0.0;
			Type alphaGrad = -(0.5) * smooth_term(mean, mean);
			sigma2 = 0.0;

			for (uint16_t i = 0; i < dataSize; i++) {
				const Type psi = lambda + alpha * eigenvalue[i];
				const Type chi = inv_sigma2 + psi;

				lambdaGrad += -(0.5) * mean[i] * mean[i] + (0.5) * inv_sigma2 / (chi * psi);
				alphaGrad += (0.5) * eigenvalue[i] * inv_sigma2 / (chi * psi);
				for (int k = 0; k < enumerate; k++) {
					sigma2 += std::pow(noise[k][i] - mean[i], 2) + 1 / chi;
				}
			}

			lambdaGrad /= dataSize * enumerate;
			alphaGrad /= dataSize * enumerate;

			// �p�����[�^�̍X�V
			this->sigma2 /= dataSize * enumerate;
			this->lambda += lambdaRate * lambdaGrad;
			this->alpha += alphaRate * alphaGrad;
		}

		py::array_t<Type> processBlock(const py::array_t<Type> nd_noise)
		{
			matrix noise = ndarray_to_vector(nd_noise);

			setData(noise);
			this->avgData = averaged(noise);
			vec mean = avgData;

			this->gs(avgData, mean);
			this->predParam(data, mean);

			for (epoch = 0; epoch < maxepoch; epoch++) {
				this->gs(avgData, mean);
				if (error < eps) {
					break;
				}
				this->predParam(data, mean);
			}
#ifdef DEBUG_MODE
			std::cout << "lambda : " << lambda << std::endl;
			std::cout << "alpha : " << alpha << std::endl;
			std::cout << "sigma2 : " << sigma2 << std::endl;
			std::cout << "error/ ep=" << epoch << " :" << error << std::endl;
#endif	// DEBUG_MODE

			py::array_t<Type> result = vector_to_ndarray(decenterize(mean));
			return result;
		}

		// =================================================================
		// accessor
		Type getLambda() const { return lambda; }
		void setLambda(const Type _lambda) { this->lambda = static_cast<Type>(_lambda); }
		Type getAlpha() const { return alpha; }
		void setAlpha(Type _alpha) { this->alpha = static_cast<Type>(_alpha); }
		Type getSigma2() const { return sigma2; }
		void setSigma2(const Type _sigma2) { this->sigma2 = static_cast<Type>(_sigma2); }
		void setMaxEpoch(int _maxepoch){ this->maxepoch = _maxepoch; }
		void setEps(const Type _eps) { this->eps = _eps; }
		Type getError() const { return static_cast<Type>(this->error); }
		void setLambdaRate(const Type _lambdaRate) { this->lambdaRate = static_cast<Type>(_lambdaRate); }
		Type getLambdaRate() { return this->lambdaRate; }
		void setAlphaRate(const Type _alphaRate) { this->alphaRate = static_cast<Type>(_alphaRate); }
		Type getAlphaRate() { return this->alphaRate; }
		int getEpoch() { return epoch; }
		vec getAvgData() { return avgData; }

	private:
		// =================================================================
			// GMRF�p�����[�^
		int enumerate;
		Type lambda;
		Type alpha;
		Type sigma2;

		// ����p�����[�^
		int maxepoch;
		Type eps;
		Type error = 0.0;
		Type lambdaRate;
		Type alphaRate;
		Type dataExpect;

		// �f�[�^�T�C�Y
		int dataSize;

		// �f�[�^�ێ��ϐ�
		vec eigenvalue;
		vec avgData;
		matrix data;

		// �����f�[�^
		int epoch;

		// =================================================================
			// �O���t���v���V�A���̌ŗL�l
		vec calcEigenVal()
		{
			vec _eigenvalue = vecInit();
			for (uint16_t i = 0; i < dataSize; i++)
			{
				_eigenvalue[i] = 4 * std::pow(std::sin(0.5 * M_PI * i / dataSize), 2);
			}
			return _eigenvalue;
		}

		// �������򉻉摜�̕��ω�
		// this->corrupted(vec)
		vec averaged(const matrix& _data)
		{
			vec _avgData = vecInit();
			for (uint16_t k = 0; k < enumerate; k++) {
				for (uint16_t i = 0; i < dataSize; i++) {
					_avgData[i] += _data[k][i] / enumerate;
				}
			}
			return _avgData;
		}

		// ���Ғl�̌v�Z
		Type calcExpect(const vec& _data)
		{
			Type _dataExpect = 0.0;
			for (uint16_t i = 0; i < dataSize; i++)
			{
				_dataExpect += _data[i];
			}
			return dataExpect /= dataSize;
		}

		// �f�[�^�̒��S��
		// this->denoised(vec)
		vec centerize(vec &_data)
		{
			vec centered = vecInit();
			for (uint16_t i = 0; i < dataSize; i++)
			{
				centered[i] = _data[i] - dataExpect;
			}
			return centered;
		}

		// �����f�[�^�̒��S��
		// this->data(matrix)
		matrix centerize2D(matrix _data)
		{
			// �f�[�^�̊��Ғl���v�Z
			this->dataExpect = calcExpect(averaged(_data));

			// �e�f�[�^�𒆐S������
			for (int k = 0; k < enumerate; k++) {
				_data.at(k) = centerize(_data.at(k));
			}
			return _data;
		}

		// ���S������
		vec decenterize(vec _data)
		{
			for (uint16_t i = 0; i < dataSize; i++) {
				_data[i] += dataExpect;
			}
			return _data;
		}

		// x^T * �� * y �̌v�Z
		Type smooth_term(const vec& x, const vec& y)
		{
			Type tmp = 0.0;
			for (uint16_t i = 0; i < this->dataSize; i++) {
				if (i + 1 < dataSize) {
					tmp += std::pow(x[i] + y[i + 1], 2);
				}
			}
			return tmp;
		}

		// 1����vector�̏�����
		vec vecInit()
		{
			vec tmp(dataSize);
			return tmp;
		}
	};
}
