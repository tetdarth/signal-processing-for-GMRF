import wave_utillity as wutil
from gmrf.x64.Release import gmrf
import numpy as np

w = wutil.wave_utillity()
w.fs = 128
w.update(info=False)

# 波形の生成
freqs = [(4, 1.0), (9, 1.0), (16, 1.0), (25, 1.0)]
wave = w.create_wave(freqs)
w.wave_plot(wave, savefig=True, title="original_wave")
w.freq_plot(w.fft(wave, window=True), savefig=True, title="original_freq")

# 画像の劣化
corrupted = w.wave_corruption(wave, 0, 1.2)
corrupted1 = w.wave_corruption(wave, 0, 1.5)
corrupted2 = w.wave_corruption(wave, 0, 1.8)

# 劣化信号をまとめたndarray
cor_waves = [corrupted,     
             corrupted1,
             corrupted2,
             ]

avg = w.wave_avg(cor_waves)     # 平均化信号

# ivgmrfによるノイズ除去
ivgmrf = gmrf.ivgmrf.ivgmrf()
ivgmrf._lambda = 1e-11
ivgmrf._lambda_rate = 1e-11
ivgmrf._alpha_rate = 1e-8
ivgmrf._epoch = 1000
ivgmrf.set_eps(1e-11)
denoised_for_ivgmrf = ivgmrf.denoise(cor_waves)     # ノイズ除去信号
print("[ivgmrf] iter = {}".format(ivgmrf._epoch))
print("[ivgmrf] predict sigma = {}".format(ivgmrf._sigma2))
print(" ============== ")

# dvgmrfによるノイズ除去
dvgmrf = gmrf.dvgmrf.dvgmrf()
dvgmrf._lambda = 1e-11
dvgmrf._lambda_rate = 1e-11
dvgmrf._alpha_rate = 1e-8
dvgmrf._epoch = 1000
dvgmrf.set_eps(1e-11)
denoised_for_dvgmrf = dvgmrf.denoise(cor_waves)     # ノイズ除去信号
print("[dvgmrf] iter = {}".format(dvgmrf._epoch))
print("[dvgmrf] predict sigma = {}".format(dvgmrf._sigma2))
print(" ============== ")

# ivhgmrfによるノイズ除去
ivhgmrf = gmrf.ivhgmrf.ivhgmrf()
ivhgmrf._lambda = 1e-5
ivhgmrf._lambda_rate = 1e-8
ivhgmrf._alpha = 1e-6
ivhgmrf._alpha_rate = 1e-6
ivhgmrf._gamma2 = 1e-6
ivhgmrf._gamma2_rate = 1e-6
ivhgmrf._epoch = 500
ivhgmrf.set_eps(1e-306)
denoised_for_ivhgmrf = ivhgmrf.denoise(cor_waves)
print("[ivhgmrf] iter = {}".format(ivhgmrf._epoch))
print("[ivhgmrf] predict sigma = {}".format(ivhgmrf._sigma2))

w.wave_compare([(wave, "original"),
                # (avg, "average"),
                # (denoised_for_ivgmrf, "ivgmrf"),
                (denoised_for_dvgmrf, "dvgmrf"),
                # (denoised_for_ivhgmrf, "ivhgmrf")
                ],
                view = True,
                savefig=False,
                title = "gmrf compare - wave"
               )

w.freq_compare([(w.fft(wave), "original"),
                (w.fft(avg), "average"),
                (w.fft(denoised_for_ivgmrf), "ivgmrf"),
                (w.fft(denoised_for_dvgmrf), "dvgmrf"),
                # (w.fft(denoised_for_ivhgmrf), "ivhgmrf"),
                ],
                view = True,
                savefig=False,
                title = "gmrf compare - freq"
                )
