import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit
import time
import glob
import shutil
from enum import Enum, auto
from gmrf.x64.Release import gmrf
from cutill.x64.Release import cutill as c


# 実行時間表示
show_time = True

'''################ parameters #####################'''
# データセットのパス
# data_path = data_path.H002.fl_center.value

# データを切り出すパラメータ
fs = 128        # サンプリング周波数
frame_time = 10     # 窓サイズ (10秒)
interval_time = 1    # スライス間隔  (4秒)
frame = frame_time*fs   # 窓幅
interval = interval_time*fs    # スライス幅
df = fs/frame   # 1サンプルあたりの周波数間隔
han = np.hanning(frame)     # ハン窓
acf = frame/sum(han)    # ハン窓の面積比

# 不整合データのパラメータ
tolerance_max = 4000.0
tolerance_min = 96.0

'''################ utillity #####################'''
# ノイズ除去クラスの列挙型
class denoise(Enum):
    CMN = auto()
    GMRF = auto()

# 配列が正常値内に存在していればTrue
def is_tolerance(data):
    if not tolerance_min <= np.min(data) or not np.max(data) <= tolerance_max:
        return False
    return True

# 波形を簡易プロット
def wave_plot(wave1, wave2=None, title=None):
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
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        # plt.xlim(0, 128)

    if title is not None:
        plt.title(title)
    plt.title("Waveform")

    plt.show()

# 周波数の簡易プロット
def freq_plot(fft_result1, fft_result2, sampling_rate, 
              title=None, 
              legend=[None, None]
    ) -> None:
    """
    2つのFFT結果を重ねてプロットする関数
    
    Parameters:
    - fft_result1: numpy array, FFT結果1
    - fft_result2: numpy array, FFT結果2
    - sampling_rate: int, サンプリング周波数（Hz）
    """
    # nをそれぞれのFFT結果の配列サイズから取得
    n1 = len(fft_result1)
    n2 = len(fft_result2)

    # 周波数軸を計算（最大長のFFT結果に合わせる）
    n = max(n1, n2)
    freqs = np.arange(0, sampling_rate, 1/(n1//sampling_rate))
    
    # 周波数軸と振幅の片側スペクトル
    half_n = n // 2
    freqs = freqs[:half_n]
    amplitude1 = fft_result1[:half_n]
    amplitude2 = fft_result2[:half_n]
    
    # プロット
    plt.figure(figsize=(10, 6))
    for i in range(2):
        if legend[i] is None:
            legend[i] = f'FFT Result{i+1}'
    plt.plot(freqs, amplitude1, label=legend[0], color="blue")
    plt.plot(freqs, amplitude2, label=legend[1], color="red", linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    if title is None:
        plt.title("Comparison of Two FFT Results")
    else:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def abs_error(wave1, wave2):
    return np.sum(np.abs(wave1 - wave2))

# 対数振幅スペクトルに変換する前に、0以下の値を無視する
def log_magnitude_spectrum(spectrum):
    spectrum[spectrum <= 0] = np.mean(spectrum)  # 0以下の値を最小の正の値に置き換える
    return np.log(np.abs(spectrum)) * 20

def fourth_dim(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e

def fit(data):
    x = np.arange(len(data))
    popt, _ = curve_fit(fourth_dim, x, data)
    return fourth_dim(x, *popt)

'''################# preprocess ####################'''
# 前処理
def slicer(dir):
    # pandasでcsvを読み込み
    wave = pd.read_csv(".."/dir/"wave.csv", names=["L", "R", "L_gain", "R_gain"])
    posture = pd.read_csv(".."/dir/"position.csv")

    # waveの長さ [s]
    wave_time = int(len(wave)/fs)
    rdata = np.empty((0, frame))   # rightの最終的な配列
    ldata = np.empty((0, frame))   # leftの最終的な配列
    pdata = np.empty(0)            # positionの最終的な配列を格納するndarray
    i = 0

    # rawデータの切り出し
    for start in range(0, wave_time-frame_time, interval_time):
        i += 1
        # waveとpositionを4秒間隔で10秒間スライス
        window = wave[start*fs : (start + frame_time)*fs]
        pos = posture[start : start + frame_time]

        # waveとをnumpy配列に変換
        lraw = window['L'].to_numpy()
        rraw = window['R'].to_numpy()
        lgain = window['L_gain'].to_numpy()
        rgain = window['R_gain'].to_numpy()
        pos = pos.to_numpy()

        # データの整合性を確認
        if not c.is_identical_element(rgain) or not c.is_identical_element(lgain):
            continue
        if not is_tolerance(rraw) or not is_tolerance(lraw):
            continue
        if not c.is_identical_element(pos):
            continue
        if np.any(pos == 0):
            continue

        # 波形の復元
        left = lraw * 2.818 ** lgain
        right = rraw * 2.818 ** rgain
        
        left = left.astype(np.float32)
        right = right.astype(np.float32)

        ldata = np.vstack((ldata, left)) if ldata.size else left
        rdata = np.vstack((rdata, right)) if rdata.size else right
        pdata = np.append(pdata, pos[0])

    print(f"{dir} | data[{len(pdata)} / {i}]")
    return ldata, rdata, pdata

'''################# CMN ####################'''
# CMNによる特徴量抽出
def cmn_denoise(ldata, rdata, concat=True):
    if concat:
        cdata = np.empty((0, 100), dtype=np.float32)   # ケプストラムの最終的な配列を格納するndarray
    else:
        cdata = []  # タプルを格納するリスト

    # CMNを適用
    final_ldata = np.empty((0, frame))
    final_rdata = np.empty((0, frame))

    for left, right in zip(ldata, rdata):
        # 波形の正規化
        left, right = c.normalize(left, right)
        
        # 窓関数を適用
        left *= han
        right *= han

        # FFT
        left_freq = fft(left)
        right_freq = fft(right)

        # 対数振幅スペクトルに変換
        left_freq = np.log(np.abs(left_freq)) * 20
        right_freq = np.log(np.abs(right_freq)) * 20

        # 低周波を除去
        left_freq[0:11] *= 1e-10
        right_freq[frame-10:] *= 1e-10

        # ケプストラムに変換
        left_cep = ifft(left_freq).real
        right_cep = ifft(right_freq).real

        # ケプストラムに変換したデータをスタック
        final_ldata = np.vstack((final_ldata, left_cep)) if final_ldata.size else left_cep
        final_rdata = np.vstack((final_rdata, right_cep)) if final_rdata.size else right_cep

    # 平均正規化
    left_cep_mean = np.mean(final_ldata, axis=0)
    right_cep_mean = np.mean(final_rdata, axis=0)

    for left_cep, right_cep in zip(final_ldata, final_rdata):
        # 平均ベクトルを減算
        left_cep -= left_cep_mean
        right_cep -= right_cep_mean

        # 左右の周波数を結合
        if concat:
            cep = np.hstack((left_cep[:50], right_cep[frame-50:]))
            # dataを2次元numpy配列として追加
            cdata = np.vstack((cdata, cep)) if cdata.size else cep
        else:
            cep = (left_cep[:50], right_cep[frame-50:])
            cdata.append(cep)

    return np.array(cdata) if concat else cdata


'''################# fitting ####################'''
def fit_deconv(ldata, rdata, concat=False):
    if concat:
        cdata = np.empty((0, 100), dtype=np.float32)   # ケプストラムの最終的な配列を格納するndarray
    else:
        cdata = []  # タプルを格納するリスト

    final_ldata = np.empty((0, frame))
    final_rdata = np.empty((0, frame))

    for left, right in zip(ldata, rdata):
        # 波形の正規化
        left, right = c.normalize(left, right)
        
        # 窓関数を適用
        left = left * han
        right = right * han

        # FFT
        left_freq = fft(left, norm="ortho")
        right_freq = fft(right, norm="ortho")

        # 対数振幅スペクトルに変換
        left_freq = np.log(np.abs(left_freq)) * 20
        right_freq = np.log(np.abs(right_freq)) * 20

        # 四次関数フィッティング
        # left_freq -= fit(left_freq)
        # right_freq -= fit(right_freq)

        # 低周波を除去
        left_freq[0:11] *= 1e-10
        right_freq[frame-10:] *= 1e-10

        # ケプストラムに変換したデータをスタック
        final_ldata = np.vstack((final_ldata, left_freq)) if final_ldata.size else left_freq
        final_rdata = np.vstack((final_rdata, right_freq)) if final_rdata.size else right_freq

    for left, right in zip(final_ldata, final_rdata):
        # 左右の周波数を結合
        if concat:
            data = np.hstack((left[frame//2:], right[frame//2:]))
            # dataを2次元numpy配列として追加
            cdata = np.vstack((cdata, data)) if cdata.size else data
        else:
            data = (left[:frame//2], right[frame//2:])
            cdata.append(data)

    return np.array(cdata) if concat else cdata


'''################# GMRF ####################'''
def gmrf_denoise(ldata, rdata):
    fdata = np.empty((0, frame))   # スペクトルの最終的な配列
    lg = gmrf.dvgmrf.dvgmrf()
    rg = gmrf.dvgmrf.dvgmrf()

    for left, right in zip(ldata, rdata):
        # 波形の正規化
        left, right = c.normalize(left, right)

        # 補正 [0, 255]
        left = (left + 1) * 127.5
        right = (right + 1) * 127.5

        # GMRFのハイパパラメータを初期化
        lg._lambda = 1e-7
        lg._lambda_rate = 1e-8
        lg._alpha = 0.2
        lg._alpha_rate = 1e-6
        lg._epoch = 1000
        lg.set_eps(1e-6)

        rg._lambda = 1e-7
        rg._lambda_rate = 1e-8
        rg._alpha = 0.2
        rg._alpha_rate = 1e-6
        rg._epoch = 1000
        rg.set_eps(1e-6)

        # ノイズ除去 [-1, 1]
        denoised_left = lg.denoise([left]) / 127.5 - 1
        denoised_right = rg.denoise([right]) / 127.5 - 1
        left = left / 127.5 - 1
        right = right / 127.5 - 1

        wave_plot(left, denoised_left)
        print(abs_error(left, denoised_left))

        # ノイズ除去に失敗したら処理を飛ばす
        if np.isnan(lg._sigma2) or np.isnan(rg._sigma2):
            # print("denoising failed")
            continue
        if abs_error(left, denoised_left) > 1 or abs_error(right, denoised_right) > 1:
            # print("denoising failed")
            continue

        # 窓関数の適用
        denoised_left = denoised_left * han
        denoised_right = denoised_right * han

        # fft
        left_freq = fft(denoised_left, norm="ortho")
        right_freq = fft(denoised_right, norm="ortho")

        # 対数振幅スペクトルに変換
        left_freq = np.log(np.abs(left_freq)) * 20
        right_freq = np.log(np.abs(right_freq)) * 20

        freq_plot(left_freq)

        freq = np.hstack((left_freq[10:frame//2], right_freq[frame//2:frame-10]))
        fdata = np.vstack((fdata, freq)) if fdata.size else freq

    return fdata


# データセットの作成
def create_dataset(dir):
    if show_time:
        t1 = time.time()   # 時間計測start

    # ディレクトリ内のcsvファイルを削除
    for p in glob.glob(".\\datasets\\" + dir +"\\***", recursive=True):
        if os.path.isdir(p):
            shutil.rmtree(p)

    # rawデータの前処理
    left, right, posture = slicer("raw\\"+dir)
    freq = gmrf_denoise(left, right)
    files = {1:0, 2:0, 3:0, 4:0,}

    # データをcsvに書き出し
    for i in range(len(freq)):
        # ndarrayをDataFrameに変換
        fdata = pd.DataFrame({'data':freq[i]}).astype(np.float64)
        pdata = int(posture[i])
        files[pdata] += 1

        # DataFrameをcsvに書き出し
        path = ".\\datasets\\" + dir +"\\" + str(pdata)
        if not os.path.exists(path):
            os.makedirs(path)
        fdata.to_csv(path + "\\" + str(files[pdata]) + ".csv")

    if show_time:
        t2 = time.time()    # 時間計測end
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time:.3}[s]")

'''
path = dpath.get_path(dpath.type.LMH, dpath.testers.H002, dpath.mattresses.fl_center)
left, right, posture = slicer(path[0])
freq = gmrf_denoise(left, right)
'''