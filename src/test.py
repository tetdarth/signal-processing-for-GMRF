import wave_utillity as wutil
from gmrf.x64.Release import gmrf

w = wutil.wave_utill()
w.set_fs(128)

freqs = [(5, 0.3), (12, 1.3), (26, 0.8), (17, 0.8)]
wave = w.create_wave(freqs, plot = True)
w.fft(wave, plot = True)

corrupted = w.wave_corruption(wave, 0, 0.8, plot = True)
w.fft(corrupted, plot = True)

m = gmrf.ivhgmrf.ivhgmrf()
m.lambda_rate = 1e-10
m.alpha_rate = 1e-9
denoised = m.denoise([corrupted])
w.wave_plot(denoised, title = "denoised for gmrf")
w.fft(denoised, plot = True)

