import gmrf
import numpy as np
import matplotlib.pyplot as plt

filename = "sine_440hz.wave"

a = 1     # 振幅
f = 12.0     # 周波数
sec = 1.0   # 信号の長さs
fs = 128    # サンプリング周波数

rng_mt = np.random.Generator(np.random.MT19937())

t = np.arange(0, sec, 1/fs)
y = a * np.sin(2 * np.pi * f * t) + rng_mt.normal(0, 0.3, len(t))
plt.ylim(-1.8, 1.8)
plt.plot(t, y)
plt.show()

### FFT: tの関数をfの関数にする ###
N = int(fs * sec)
y_fft = np.fft.fft(y) # 離散フーリエ変換
freq = np.fft.fftfreq(N, d=(1/fs)) # 周波数を割り当てる（※後述）
Amp = abs(y_fft/(N/2)) # 音の大きさ（振幅の大きさ）
plt.ylim(0, 1.2)
plt.plot(freq[1:int(N/2)], Amp[1:int(N/2)]) # A-f グラフのプロット
# plt.xscale("log") # 横軸を対数軸にセット
plt.show()

m = gmrf.ivgmrf()
m.set_eps(0.000000001)
m.epoch = 10000
m.alpha_rate = 0.0000001
m.lambda_rate = 0.000000001
u = m.denoise([y])
print(m.epoch)
plt.ylim(-1.8, 1.8)
plt.plot(t, u)
plt.show()


### FFT: tの関数をfの関数にする ###
N = int(fs * sec)
y_fft = np.fft.fft(u) # 離散フーリエ変換
freq = np.fft.fftfreq(N, d=(1/fs)) # 周波数を割り当てる（※後述）
Amp = abs(y_fft/(N/2)) # 音の大きさ（振幅の大きさ）
plt.ylim(0, 1.2)
plt.plot(freq[1:int(N/2)], Amp[1:int(N/2)]) # A-f グラフのプロット
# plt.xscale("log") # 横軸を対数軸にセット
plt.show()

fig, ax = plt.subplots()
ax.plot(t, y)
ax.plot(t, u)
plt.show()