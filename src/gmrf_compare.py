import wave_utillity as wutil
from gmrf.x64.Release import gmrf
import numpy as np

w = wutil.wave_utillity()
w.fs = 128
w.update(info=False)

# 波形の生成
freqs = [(6, 0.4), (9, 0.6), (14, 0.6)]
wave = w.create_wave(freqs)
w.wave_plot(wave, savefig=True, title="original_wave", window=False)
w.freq_plot(w.fft(wave, window=True), savefig=True, title="original_freq")

# 画像の劣化
corrupted = w.wave_corruption(wave, 0, 0.2)
corrupted1 = w.wave_corruption(wave, 0, 0.3)
corrupted2 = w.wave_corruption(wave, 0, 0.25)

# 劣化信号をまとめたndarray
cor_waves = [corrupted,     
             corrupted1,
             corrupted2,
             ]

avg = w.wave_averaging(cor_waves)     # 平均化信号
w.wave_plot(avg, title="average", savefig=True)

# ivgmrfによるノイズ除去
ivgmrf = gmrf.ivgmrf.ivgmrf()
ivgmrf._lambda = 1e-8
ivgmrf._lambda_rate = 1e-8
ivgmrf._alpha_rate = 1e-6
ivgmrf._epoch = 1000
ivgmrf.set_eps(1e-8)
denoised_for_ivgmrf = ivgmrf.denoise(cor_waves)     # ノイズ除去信号
print("[ivgmrf] iter = {}".format(ivgmrf._epoch))
print("[ivgmrf] predict sigma = {}".format(ivgmrf._sigma2))
print(" ============== ")

# dvgmrfによるノイズ除去
dvgmrf = gmrf.dvgmrf.dvgmrf()
dvgmrf._lambda = 1e-8
dvgmrf._lambda_rate = 1e-8
dvgmrf._alpha_rate = 1e-6
dvgmrf._epoch = 1000
dvgmrf.set_eps(1e-8)
denoised_for_dvgmrf = dvgmrf.denoise(cor_waves)     # ノイズ除去信号
print("[dvgmrf] iter = {}".format(dvgmrf._epoch))
print("[dvgmrf] predict sigma = {}".format(dvgmrf._sigma2))
print(" ============== ")

# ivhgmrfによるノイズ除去
ivhgmrf = gmrf.ivhgmrf.ivhgmrf()
ivhgmrf._lambda = 1e-6
ivhgmrf._lambda_rate = 5e-6
ivhgmrf._alpha_rate = 1e-6
ivhgmrf._alpha_rate = 1e-5
ivhgmrf._gamma2 = 1e-3
ivhgmrf._gamma2_rate = 1e-5
ivhgmrf._epoch = 1000
ivhgmrf.set_eps(1e-5)
denoised_for_ivhgmrf = ivhgmrf.denoise(cor_waves)
print("[ivhgmrf] iter = {}".format(ivhgmrf._epoch))
print("[ivhgmrf] predict sigma2 = {}".format(ivhgmrf._sigma2))
print("[ivhgmrf] predict alpha = {}".format(ivhgmrf._alpha))
print("[ivhgmrf] predict lambda = {}".format(ivhgmrf._lambda))
print("[ivhgmrf] predict gamma2 = {}".format(ivhgmrf._gamma2))
print(" ============== ")

# dvhgmrfによるノイズ除去
dvhgmrf = gmrf.dvhgmrf.dvhgmrf()
dvhgmrf._lambda = 1e-6
dvhgmrf._lambda_rate = 5e-6
dvhgmrf._alpha_rate = 1e-6
dvhgmrf._alpha_rate = 1e-5
dvhgmrf._gamma2 = 1e-3
dvhgmrf._gamma2_rate = 1e-5
dvhgmrf._epoch = 1000
dvhgmrf.set_eps(1e-5)
denoised_for_dvhgmrf = dvhgmrf.denoise(cor_waves)
print("[dvhgmrf] iter = {}".format(dvhgmrf._epoch))
print("[dvhgmrf] predict sigma = {}".format(dvhgmrf._sigma2))
print("[dvhgmrf] predict alpha = {}".format(dvhgmrf._alpha))
print("[dvhgmrf] predict lambda = {}".format(dvhgmrf._lambda))
print("[dvhgmrf] predict gamma2 = {}".format(dvhgmrf._gamma2))
print(" ============== ")

w.wave_compare([(wave, "original"),
                (avg, "average"),
                (denoised_for_ivgmrf, "ivgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - wave - ivgmrf"
               )

w.wave_compare([(wave, "original"),
                (avg, "average"),
                (denoised_for_dvgmrf, "dvgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - wave - dvgmrf"
               )

w.wave_compare([(wave, "original"),
                (avg, "average"),
                (denoised_for_ivhgmrf, "ivhgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - wave - ivhgmrf"
               )

w.wave_compare([(wave, "original"),
                (avg, "average"),
                (denoised_for_dvhgmrf, "dvhgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - wave - dvhgmrf"
               )

w.freq_compare([(w.fft(wave), "original"),
                (w.fft(avg), "average"),
                (w.fft(denoised_for_ivgmrf), "ivgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - freq - ivgmrf"
                )

w.freq_compare([(w.fft(wave), "original"),
                (w.fft(avg), "average"),
                (w.fft(denoised_for_dvgmrf), "dvgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - freq - dvgmrf"
                )

w.freq_compare([(w.fft(wave), "original"),
                (w.fft(avg), "average"),
                (w.fft(denoised_for_ivhgmrf), "ivhgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - freq - ivhgmrf"
                )

w.freq_compare([(w.fft(wave), "original"),
                (w.fft(avg), "average"),
                (w.fft(denoised_for_dvhgmrf), "dvhgmrf"),
                ],
                view = True,
                savefig=True,
                title = "gmrf compare - freq - dvhgmrf"
                )
