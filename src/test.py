import wave_utillity as wutil
from gmrf.x64.Release import gmrf

w = wutil.wave_utillity()
w.fs = 128
w.update(info=False)

# 波形の生成
freqs = [(5, 1.2), (12, 0.8), (26, 0.5), (17, 0.6)]
wave = w.create_wave(freqs)

# 画像の劣化
corrupted = w.wave_corruption(wave, 0, 1.2)
corrupted1 = w.wave_corruption(wave, 0, 0.9)
corrupted2 = w.wave_corruption(wave, 0, 0.6)

# 劣化信号をまとめたndarray
cor_waves = [corrupted,     
             corrupted1,
             corrupted2,
             ]

avg = w.wave_avg(cor_waves)     # 平均化信号

# gmrfによるノイズ除去
m = gmrf.dvgmrf.dvgmrf()
m._lambda = 1e-11
m._lambda_rate = 1e-11
m._alpha_rate = 1e-8
m._epoch = 500
m.set_eps(1e-11)
denoised = m.denoise(cor_waves)     # ノイズ除去信号
print("predict iter = {}".format(m._epoch))

# 波形の時間軸での比較
w.wave_compare([(wave, "origin"), 
                # (corrupted, "corrupted"),
                # (avg, "average"), 
                (denoised, "gmrf")
                ], 
               savefig=True,
               view=True, 
               title="wave_compare"
)

# 波形の周波数軸での比較
w.freq_compare([(w.fft(wave), "origin"),
                # (w.fft(corrupted), "corrupted"),
                # (w.fft(avg), "average"),
                (w.fft(denoised), "gmrf")
                ],
               savefig=True,
               view = True,
               title = "freq_compare"
)
