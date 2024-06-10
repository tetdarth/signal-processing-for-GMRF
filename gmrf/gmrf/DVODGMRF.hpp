#pragma once

#include "include.hpp"

#define GSITER 2
// #define DEBUG_MODE

// 1�����f�[�^�ɑ΂���DVGMRF
namespace GMRF
{
	template<typename Type>
	class dvgmrf_od
	{
		typedef std::vector<Type> vec;
		typedef std::vector<std::vector<Type>> matrix;

	public:
		// =================================================================
		dvgmrf_od() {
			setLambda(1e-11);
			setAlpha(1e-8);
			setSigma2(5e-01);
			setMaxEpoch(1000);
			setEps(1e-9);
			setLambdaRate(1e-13);
			setAlphaRate(1e-6);
		}

		~dvgmrf_od() {}

		// =================================================================
		void setData(const matrix& _data)
		{
			this->enumerate = static_cast<int>(_data.size());
			this->dataSize = static_cast<int>(_data.at(0).size());
			this->sigma2 = vecInit(this->initdev);
			this->data = centerize(_data);
			this->avgData = averaged(_data);
			this->eigenvalue = calcEigenVal();
		}

		void gs(const matrix& noise, vec& mean)
		{
			Type inv_sigma2 = 0;
			for (int k = 0; k < enumerate; k++) {
				inv_sigma2 += 1 / sigma2[k];
			}

			// �K�E�X�E�U�C�f���@
			for (int iter = 0; iter < GSITER; iter++) {
				error = 0.0;

				for (uint16_t i = 0; i < dataSize; i++) {
					// ����
					Type denominator = inv_sigma2 + lambda;
					// ���q
					Type numerator = 0.0;
					for (int k = 0; k < enumerate; k++) {
						numerator += noise[k][i] / sigma2[k];
					}

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
			Type inv_sigma2 = 0;
			for (int k = 0; k < enumerate; k++) {
				inv_sigma2 += 1 / sigma2[k];
			}

			Type lambdaGrad = 0.0;
			Type alphaGrad = -(0.5) * smooth_term(mean, mean);

			for (int k = 0; k < enumerate; k++) {
				for (int i = 0; i < dataSize; i++) {
					const Type psi = lambda + alpha * eigenvalue[i];
					const Type chi = inv_sigma2 + psi;

					lambdaGrad += -(0.5) * mean[i] * mean[i] + (0.5) * inv_sigma2 / (chi * psi);
					alphaGrad += (0.5) * eigenvalue[i] * inv_sigma2 / (chi * psi);
					sigma2[k] += std::pow(noise[k][i] - mean[i], 2) + 1 / chi;
				}
				sigma2[k] /= dataSize;
			}

			lambdaGrad /= dataSize * enumerate;
			alphaGrad /= dataSize * enumerate;

			// �p�����[�^�̍X�V
			this->lambda += lambdaRate * lambdaGrad;
			this->alpha += alphaRate * alphaGrad;
		}

		vec processBlock(const matrix& noise)
		{
			setData(noise);
			vec mean = avgData;

			gs(this->data, mean);
			predParam(this->data, mean);

			for (epoch = 0; epoch < maxepoch; epoch++) {
				gs(data, mean);
				if (error < eps) {
					break;
				}
				predParam(data, mean);
			}
#ifdef DEBUG_MODE
			std::cout << "lambda : " << lambda << std::endl;
			std::cout << "alpha : " << alpha << std::endl;
			std::cout << "sigma2[0] : " << sigma2[0] << std::endl;
			std::cout << "sigma2[1] : " << sigma2[1] << std::endl;
			std::cout << "sigma2[2] : " << sigma2[2] << std::endl;
			std::cout << "error/ ep=" << epoch << " :" << error << std::endl;
#endif	// DEBUG_MODE

			return decenterize(mean);
		}

		// =================================================================
		// accessor
		Type getLambda() const { return lambda;}
		void setLambda(const Type _lambda) { this->lambda = static_cast<Type>(_lambda);}
		Type getAlpha() const { return alpha;}
		void setAlpha(Type _alpha) { this->alpha = static_cast<Type>(_alpha);}
		Type getSigma2() const { return sigma2;}
		void setSigma2(const Type _sigma2) { this->initdev = static_cast<Type>(_sigma2); }
		void setMaxEpoch(int _maxepoch) { this->maxepoch = _maxepoch; }
		void setEps(const Type _eps) { this->eps = _eps; }
		Type getError() const { return static_cast<Type>(this->error);}
		void setLambdaRate(const Type _lambdaRate) { this->lambdaRate = static_cast<Type>(_lambdaRate); }
		void setAlphaRate(const Type _alphaRate) { this->alphaRate = static_cast<Type>(_alphaRate); }
		int getEpoch() const { return epoch; }
		vec getAvgData() const { return avgData; }

	private:
		// =================================================================
		// GMRF�p�����[�^
		int enumerate;
		Type lambda;
		Type alpha;
		vec sigma2;
		Type initdev;

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
			vec _eigenvalue(dataSize, 0.0);
			for (int i = 0; i < dataSize; i++)
			{
				_eigenvalue[i] = 4 * std::pow(std::sin(0.5 * M_PI * i / dataSize), 2);
			}
			return _eigenvalue;
		}

		// �������򉻉摜�̕��ω�
		// this->corrupted(vec)
		vec averaged(const matrix& src)
		{
			int _enumerate = static_cast<int>(src.size());
			int _dataSize = static_cast<int>(src.at(0).size());
			vec _averaged(_dataSize, 0.0);

			for (int k = 0; k < _enumerate; k++) {
				for (int i = 0; i < _dataSize; i++) {
					_averaged[i] += src[k][i] / _enumerate;
				}
			}
			return _averaged;
		}

		// ���Ғl�̌v�Z
		Type calcExpect(const vec& _data)
		{
			Type dataEx = 0.0;
			for (int i = 0; i < _data.size(); i++)
			{
				dataEx += _data[i];
			}
			return dataEx /= _data.size();
		}

		// �f�[�^�̒��S��
		// this->denoised(vec)
		matrix centerize(const matrix& _data)
		{
			int _enumerate = static_cast<int>(_data.size());
			int _dataSize = static_cast<int>(_data.at(0).size());
			matrix _centered(_enumerate, std::vector<Type>(_dataSize, 0.0));
			this->dataExpect = calcExpect(averaged(_data));

			for (int k = 0; k < _enumerate; k++) {
				for (int i = 0; i < _dataSize; i++) {
					_centered[k][i] = _data[k][i] - dataExpect;
				}
			}
			return _centered;
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
		vec vecInit(Type initValue)
		{
			vec tmp(enumerate, initValue);
			return tmp;
		}
	};
}