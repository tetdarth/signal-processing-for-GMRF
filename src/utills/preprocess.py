import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit
from enum import Enum, auto
from gmrf.x64.Release import gmrf
from cutill.x64.Release import cutill as c
from scipy.signal import firwin, lfilter


# 実行時間表示
show_time = True

'''################ parameters #####################'''
# データセットのパス
# data_path = data_path.H002.fl_center.value

# データを切り出すパラメータ
fs = 128        # サンプリング周波数
frame_time = 10     # 窓サイズ (10秒)
interval_time = 4    # スライス間隔  (4秒)
frame = frame_time*fs   # 窓幅
interval = interval_time*fs    # スライス幅
df = fs/frame   # 1サンプルあたりの周波数間隔
han = np.hanning(frame)     # ハン窓

# 呼吸情報のカットオフ周波数
breath_cutoff = 1.0

# 不整合データのパラメータ
tolerance_max = 4000.0
tolerance_min = 96.0

# CMNのパラメータ
quef_high_cutoff = 50
quef_low_cutoff = 0

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
def wave_plot(wave1, wave2=None, title=None, fs=128):
    sample = np.arange(0, len(wave1)/fs, 1/fs)
    plt.figure(figsize=(6, 4))
    if wave2 is None:
        plt.plot(sample, wave1)
        plt.title("Waveform")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [dB]")
    else:
        plt.plot(sample, wave1, label="Left")
        plt.plot(sample, wave2, label="Right")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [dB]")
        # plt.xlim(0, 128)

    if title is not None:
        plt.title(title)

    plt.show()

# 周波数の簡易プロット
def freq_plot(fft_result1, 
              fft_result2=None, 
              sampling_rate=fs, 
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
    n2 = len(fft_result2) if fft_result2 is not None else -1

    # 周波数軸を計算（最大長のFFT結果に合わせる）
    n = max(n1, n2)
    freqs = np.arange(0, sampling_rate, 1/(n1//sampling_rate))
    
    # 周波数軸と振幅の片側スペクトル
    half_n = n // 2
    freqs = freqs[:half_n]
    amplitude1 = fft_result1[:half_n]
    if n2 > 0:
        amplitude2 = fft_result2[:half_n]
    
    # プロット
    plt.figure(figsize=(10, 6))
    for i in range(2):
        if legend[i] is None:
            legend[i] = f'FFT Result{i+1}'
    plt.plot(freqs, amplitude1, label=legend[0], color="blue")
    if n2 > 0:
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

def fit_polynomial(data, order):
    """
    指定された次数でデータを近似する多項式を生成

    Parameters:
    data (array-like): フィットさせるデータ
    order (int): 多項式の次数

    Returns:
    func (callable): フィッティングした多項式関数
    fitted_values (array): データに基づくフィッティング結果
    """
    # x 軸の値を生成
    x = np.arange(len(data))
    
    # 多項式フィッティング
    coefficients = np.polyfit(x, data, order)
    
    # フィッティング関数を生成
    polynomial_func = np.poly1d(coefficients)
    
    # フィッティング結果
    fitted_values = polynomial_func(x)
    
    return fitted_values

'''################# preprocess ####################'''
# 前処理
def slicer(dir):
    """
    波形を読み込んでスライスし、不正データを除外する関数

    :Args:
    - dir(str): データが格納されているディレクトリ

    :Returns:
    - ldata((2D) ndarray): 左側のセンサデータ
    - rdata((2D) ndarray): 右側のセンサデータ
    - pdata(ndarray): センサデータに対応する寝姿勢ラベル
    """
    # pandasでcsvを読み込み
    wave = pd.read_csv(".."/dir/"wave.csv", names=["L", "R", "L_gain", "R_gain"])
    posture = pd.read_csv(".."/dir/"position.csv")

    wave_time = int(len(wave)/fs)   # waveの長さ [s]
    rdata = np.empty((0, frame))   # rightの最終的な配列
    ldata = np.empty((0, frame))   # leftの最終的な配列
    pdata = np.empty(0)            # positionの最終的な配列を格納するndarray
    i = 0   # データ数カウンタ

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
def cmn_denoise(ldata, rdata, concat=True, breathcut="fir"):
    if concat:
        cdata = np.empty((0, 100), dtype=np.float32)   # ケプストラムの最終的な配列を格納するndarray
    else:
        cdata = []  # タプルを格納するリスト

    # CMNを適用する部分にハイパスフィルタを追加
    final_ldata = np.empty((0, frame))
    final_rdata = np.empty((0, frame))

    if breathcut == "simple":
        real_breath_cutoff = int(breath_cutoff * frame_time)

    for left, right in zip(ldata, rdata):
        # 波形の正規化
        left, right = c.normalize(left, right)

        if breathcut=="fir":
            # ハイパスフィルタを適用
            left = apply_highpass_filter(left, cutoff=breath_cutoff, fs=fs)
            right = apply_highpass_filter(right, cutoff=breath_cutoff, fs=fs)

        # 窓関数を適用
        left *= han
        right *= han

        # FFT
        left_freq = fft(left)
        right_freq = fft(right)

        # 対数振幅スペクトルに変換
        left_freq = np.log(np.abs(left_freq)) * 20
        right_freq = np.log(np.abs(right_freq)) * 20

        if breathcut=="simple":
            # 低周波を除去
            left_freq[0:real_breath_cutoff] *= 1e-10
            left_freq[frame-real_breath_cutoff:] *= 1e-10
            right_freq[0:real_breath_cutoff] *= 1e-10
            right_freq[frame-real_breath_cutoff:] *= 1e-10

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
            cep = np.hstack((left_cep[:quef_high_cutoff], right_cep[frame-quef_high_cutoff:]))
            # dataを2次元numpy配列として追加
            cdata = np.vstack((cdata, cep)) if cdata.size else cep
        else:
            cep = (left_cep[quef_low_cutoff:quef_high_cutoff], right_cep[quef_low_cutoff:quef_high_cutoff])
            cdata.append(cep)

    return np.array(cdata) if concat else cdata

'''################ FIRフィルタ ##################'''
def apply_highpass_filter(data, cutoff, fs, num_taps=101, window='hamming'):
        """
        ハイパスFIRフィルタを適用する関数

        :Parameters:
        - data: ndarray, フィルタリングする信号データ
        - cutoff: float, カットオフ周波数（Hz）
        - fs: int, サンプリング周波数（Hz）
        - num_taps: int, フィルタ係数の数（デフォルトは101）
        - window: str, 窓関数の種類（デフォルトは 'hamming'）

        :Returns:
        - filtered_data: ndarray, フィルタリング後の信号データ
        """
        # ハイパスフィルタの係数を計算
        fir_coeff = firwin(num_taps, cutoff, fs=fs, window=window, pass_zero=False)
        # フィルタを適用
        filtered_data = lfilter(fir_coeff, 1.0, data)
        return filtered_data

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
        # left_freq -= fit_polynomial(left_freq, 4)
        # right_freq -= fit_polynomial(right_freq, 4)

        # 低周波を除去
        left_freq[0:breath_cutoff] *= 1e-10
        left_freq[frame-breath_cutoff:] *= 1e-10
        right_freq[0:breath_cutoff] *= 1e-10
        right_freq[frame-breath_cutoff:] *= 1e-10

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
