import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.fft import fft
import time
import tester
import glob
import shutil
from gmrf.x64.Release import gmrf
from cutill.x64.Release import cutill as c


# 実行時間表示
show_time = True

'''################ parameters #####################'''
# データセットのパス
data_path = tester.H002.st_center.value

# データを切り出すパラメータ
fs = 128        # サンプリング周波数
frame_time = 10     # 窓サイズ (10秒)
interval_time = 4    # スライス間隔  (4秒)
frame = frame_time*fs   # 窓幅
interval = interval_time*fs    # スライス幅
df = fs/frame   # 1サンプルあたりの周波数間隔
han = np.hanning(frame)     # ハン窓
acf = 1/(sum(han)/frame)    # ハン窓の面積

# 不整合データのパラメータ
tolerance_max = 4000.0
tolerance_min = 96.0

'''################ utillity #####################'''
# 配列要素がすべて同じならTrue
def is_identical_element(data):
    # 1から始まる連番の二次元配列を作成
    rows = 10
    cols = len(data)
    template = np.tile(np.arange(0, rows), (cols, 1)).T
    for num_template in template:
        if np.array_equiv(data, num_template):
            return True
    return False

# 配列が正常値内に存在していればTrue
def is_tolerance(data):
    if not tolerance_min <= np.min(data) or not np.max(data) <= tolerance_max:
        return False
    return True

# 波形を簡易プロット
def wave_plot(wave1, wave2=None):
    sample = np.arange(len(wave1))
    if wave2 is None:
        plt.plot(sample, wave1)
        plt.title("Waveform")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
    else:
        plt.plot(sample, wave1, label="Left")
        plt.plot(sample, wave2, label="Right")
        plt.legend()
        plt.title("Waveform")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

    plt.show()

def freq_plot(fft1, fft2=None):
    # 周波数軸の作成
    freq = np.linspace(0, fs, fft1.size)

    if fft2 is None:
        plt.figure(figsize=(8, 6))
        plt.plot(freq, fft1)
        plt.title("FFT")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")

    else:
        # スペクトルをプロット
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(freq, fft1)
        plt.title("Left FFT")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")

        plt.subplot(122)
        plt.plot(freq, fft2)
        plt.title("Right FFT")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")

    plt.show()

'''################# preprocess ####################'''
# 前処理
def preprocess(dir):
    # pandasでcsvを読み込み
    wave = pd.read_csv(dir+"\\wave.csv", names=["L", "R", "L_gain", "R_gain"])
    position = pd.read_csv(dir+"\\position.csv")

    # waveの長さ [s]
    wave_time = int(len(wave)/fs)
    pdata = np.empty(0)            # positionの最終的な配列を格納するndarray
    fdata = np.empty((0, frame))   # fftの最終的な配列を格納するndarray
    i = 0
    rdata = np.empty((0, frame))
    ldata = np.empty((0, frame))

    # 前処理
    for start in range(0, wave_time-frame_time, interval_time):
        i += 1
        # waveとpositionを4秒間隔で10秒間スライス
        window = wave[start*fs : (start + frame_time)*fs]
        pos = position[start : start + frame_time]

        # waveとをnumpy配列に変換
        rraw = window['R'].to_numpy()
        lraw = window['L'].to_numpy()
        rgain = window['R_gain'].to_numpy()
        lgain = window['L_gain'].to_numpy()
        pos = pos.to_numpy()

        # データの整合性を確認
        if not is_identical_element(rgain) or not is_identical_element(lgain):
            continue
        if not is_tolerance(rraw) or not is_tolerance(lraw):
            continue
        if not is_identical_element(pos):
            continue
        if np.any(pos == 0):
            continue

        # 波形の復元
        right = (rraw * 2.818 ** rgain)
        left = (lraw * 2.818 ** lgain)

        # 波形の正規化
        right, left = c.normalize(right, left)
        # wave_plot(left, right)

        # 窓関数を適用
        right = right * han
        left = left * han

        # FFT
        right_freq = fft(right, norm="ortho")
        left_freq = fft(left, norm="ortho")

        # 振幅スペクトルに変換
        right_freq = np.log(np.abs(right_freq)) * 20
        left_freq = np.log(np.abs(left_freq)) * 20
        # freq_plot(left_freq, right_freq)

        # 左右の周波数を結合
        freq = np.hstack((left_freq[11:(frame//2)+1], right_freq[frame//2:frame-10]))
        # freq_plot(freq)

        # dataを2次元numpy配列として追加
        rdata = np.vstack((rdata, right))
        ldata = np.vstack((ldata, left))
        pdata = np.append(pdata, pos[0]) if pdata.size else pos[0]
        fdata = np.vstack((fdata, freq)) if fdata.size else freq

    print(f"data[{len(rdata)} / {i}]")
    return fdata, pdata

# データセットの作成
def create_dataset(dir):
    if show_time:
        t1 = time.time()   # 時間計測start

    # ディレクトリ内のcsvファイルを削除
    for p in glob.glob(".\\datasets\\" + dir +"\\***", recursive=True):
        if os.path.isdir(p):
            shutil.rmtree(p)

    # rawデータの前処理
    freq, position = preprocess("raw\\"+dir)
    print(freq.shape)

    # データをcsvに書き出し
    for i in range(len(freq)):
        # ndarrayをDataFrameに変換
        fdata = pd.DataFrame(freq[i]).astype(np.float64)
        pdata = position[i]

        # DataFrameをcsvに書き出し
        path = ".\\datasets\\" + dir +"\\" + str(int(pdata))
        if not os.path.exists(path):
            os.makedirs(path)
        files = sum(os.path.isfile(os.path.join(path, name)) for name in os.listdir(path))
        fdata.to_csv(path + "\\" + str(files+1) + ".csv")

    if show_time:
        t2 = time.time()    # 時間計測end
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time:.3}[s]")


create_dataset(data_path)