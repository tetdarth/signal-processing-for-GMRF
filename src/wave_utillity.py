import numpy as np
import matplotlib.pyplot as plt

class wave_utillity:
    fs = 128
    sec = 1.0
    n = int(fs * sec)
    dt = 1.0 / fs
    Frange = fs / 2.56      # 分析周波数
    time = np.arange(0, sec, dt)      # 時間軸データ
    han = np.hanning(n)     # ハン窓の取得
    acf = 1/(sum(han)/n)    # Amplitude Correction Factor(振幅補正係数)の計算
    
    plt.rcParams["font.size"] = 14

    # 信号生成
    def create_wave(self, freqs, offset = 0):
        time = np.arange(0, self.sec, self.dt)      # 時間軸データ
        wave = time * 0     # 出力信号
        for freq, gain in freqs:
            if freq > int(self.n/2) :
                print("WARNING!!! Contains signals above the Nyquist frequency : {}Hz".format(freq))
            sine = gain * np.sin(2 * np.pi * freq * time) + offset
            wave += sine
        return wave
    
    # 信号の劣化(ガウスノイズ)
    def wave_corruption(self, wave, m, stddev):
        rng = np.random.default_rng()
        rand = rng.normal(m, stddev, len(wave))
        corrupt = wave + rand
        return corrupt

    # 信号の高速フーリエ変換
    def fft(self, wave, window=True):
        if window:
            wave = wave * self.han
        fft_wave = np.fft.fft(wave)
        fft_wave = abs(fft_wave * 2 / self.n) * self.acf
        fft_wave[0] = abs(fft_wave[0] / 2)
        return fft_wave
    
    # 平均化した波形
    def wave_avg(self, waves):
        avg = self.time * 0
        for w in waves:
            avg += w
        avg /= len(waves)
        return avg   
        
    # 波形の時間軸のプロット
    def wave_plot(self, wave, title="", savefig=False, hanning=False):
        if hanning:
            wave = wave * self.han
        plt.plot(self.time, wave) # y-t グラフのプロット
        plt.xlim(0, self.sec) # 横軸に関する描画範囲指定
        plt.title(title)
        plt.xlabel("time[sec]") #横軸ラベル
        # plt.ylabel("Amplitude spectrum[V]") #縦軸ラベル
        if savefig:
            plt.savefig(title + ".jpg", format="jpg")
        plt.show() # グラフの表示
        
    # 波形の周波数軸のプロット
    def freq_plot(self, freq, title="", savefig=False):
        # 周波数軸
        Ffreq = np.fft.fftfreq(self.n, d = self.dt)
        #FFTのプロット
        plt.plot(Ffreq[0:int(self.Frange)], freq[0:int(self.Frange)]) #グラフデータの指定（x, y）
        plt.xlim(0, self.Frange) #横軸の最小値、最大値
        plt.xlabel("Frequency[Hz]") #横軸ラベル
        plt.ylabel("Amplitude spectrum[V]") #縦軸ラベル
        plt.title(title)
        if savefig:
            plt.savefig(title + ".jpg", format="jpg")
        plt.show()
            
    # 時間軸での比較
    def wave_compare(self, waves, title="", savefig=False, view=False):
        fig, ax = plt.subplots()
        ax.set_xlabel('t')  # x軸ラベル
        ax.set_ylabel('y')  # y軸ラベル
        ax.set_title(title) # グラフタイトル   
        for wave, led in waves:
            ax.plot(self.time, wave, label=led)
            
        ax.legend()
        if savefig:
            plt.savefig(title + ".jpg", format="jpg")
        
        if view:    
            plt.show()
    
    # 周波数領域での比較
    def freq_compare(self, freqs, title="", savefig=False, view=False):
        # 周波数軸
        Ffreq = np.fft.fftfreq(self.n, d = self.dt)
        fig, ax = plt.subplots()
        ax.set_xlabel('Frequency[Hz]')
        ax.set_ylabel('Ampletude spectrum[V]')
        ax.set_title(title)
        # 信号の描画      
        for freq, led in freqs:
            ax.plot(Ffreq[0:int(self.Frange)], freq[0:int(self.Frange)], label=led)    
        ax.legend() # 凡例
        # プロットの保存
        if savefig:
            plt.savefig(title + ".jpg", format="jpg")
        # プロットの可視化
        if view:    
            plt.show()
    
    # パラメータのアップデート
    def update(self, info=False):
        self.n = int(self.fs * self.sec)
        self.dt = 1.0 / self.fs
        self.Frange = self.fs / 2.56      # 分析周波数
        self.time = np.arange(0, self.sec, self.dt)      # 時間軸データ
        self.han = np.hanning(self.n)     # ハン窓の取得
        self.acf = 1/(sum(self.han)/self.n)
        if info:
            print("===== wave info =====")
            print("sampleing freq -> {}".format(self.fs))
            print("number of samples -> {}".format(self.n))
            