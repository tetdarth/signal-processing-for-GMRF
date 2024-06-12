import gmrf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 2, 3]])

m = gmrf.ivgmrf()
result = m.denoise(x)

print(result)
print(m.alpha)

A = 1.0     # 振幅
f = 11.0     # 周波数
sec = 1.0   # 信号の長さs
sf = 128    # サンプリング周波数

print("sine-wave")
rng_mt = np.random.Generator(np.random.MT19937())

t = np.arange(0, sec, 1/sf)
y = A * np.sin(2 * np.pi * f * t) + rng_mt.normal(0, 0.3, len(t))
plt.plot(t, y)
plt.show()

u = m.denoise([y])
plt.plot(t, u)
plt.show()