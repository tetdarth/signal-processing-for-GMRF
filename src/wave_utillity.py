import numpy as np
import matplotlib.pyplot as plt

class wave_utill:
    fs = 128
    sec = 1.0
    n = int(fs * sec)
    dt = 1.0 / fs
    Frange = fs / 2.05      # 分析周波数
    time = np.arange(0, sec, dt)      # 時間軸データ

    # コンストラクタ
    def __init__(self):
        print("wave : {} samples".format(self.n))
    
    # 信号生成
    def create_wave(self, freqs, offset = 0, plot = False):
        time = np.arange(0, self.sec, self.dt)      # 時間軸データ
        wave = time * 0     # 出力信号
        for freq, gain in freqs:
            sine = gain * np.sin(2 * np.pi * freq * time) + offset
            wave += sine

        if plot:
            plt.plot(time, wave) # y-t グラフのプロット
            plt.xlim(0, self.sec) # 横軸に関する描画範囲指定
            plt.show() # グラフの表示

        return wave
    
    # 信号の劣化(ガウスノイズ)
    def wave_corruption(self, wave, m, stddev, plot = False):
        rng = np.random.default_rng()
        rand = rng.normal(m, stddev, len(wave))
        corrupt = wave + rand

        if plot:
            plt.plot(self.time, corrupt) # y-t グラフのプロット
            plt.xlim(0, self.sec) # 横軸に関する描画範囲指定
            plt.show() # グラフの表示

        return corrupt

    # Fast Furier Transform
    def fft(self, wave, plot = False):
        fft_wave = np.fft.fft(wave)
        fft_wave = abs(fft_wave * 2 / self.n)
        fft_wave[0] = abs(fft_wave[0] / 2)

        # 周波数軸
        Ffreq = np.fft.fftfreq(self.n, d = self.dt)

        if plot:
            #FFTのプロット
            plt.plot(Ffreq[0:int(self.Frange)], fft_wave[0:int(self.Frange)]) #グラフデータの指定（x, y）
            plt.xlim(0, self.Frange) #横軸の最小値、最大値
            plt.xlabel("Frequency[Hz]") #横軸ラベル
            plt.ylabel("振幅") #縦軸ラベル
            plt.show()

        return fft_wave
    
    # プロット
    def wave_plot(self, wave, title = ""):
        plt.plot(self.time, wave) # y-t グラフのプロット
        plt.xlim(0, self.sec) # 横軸に関する描画範囲指定
        plt.title(title)
        plt.show() # グラフの表示
    
    ###### accsessor #######
    def set_fs(self, _fs):
        self.fs = _fs
        self.n = int(self.sec * _fs)
        self.dt = 1 / _fs
        self.Frange = _fs / 2.05
        self.time = np.arange(0, self.sec, self.dt)
        print("wave : {} samples".format(self.n))