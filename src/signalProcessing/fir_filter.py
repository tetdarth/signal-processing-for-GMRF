import numpy as np
from scipy.signal import firwin, lfilter

def create_highpass_fir_filter(num_taps, cutoff, fs, window='hamming'):
    """
    ハイパスFIRフィルタを窓関数法で作成する関数

    :Parameters:
    num_taps: int, フィルタ係数の数（奇数が推奨）
    - cutoff: float, カットオフ周波数（Hz）
    - fs: int, サンプリング周波数（Hz）
    - window: str, 窓関数の種類（デフォルトは 'hamming'）

    :Returns:
    - fir_coeff: ndarray, FIRフィルタの係数
    """
    # ハイパスフィルタの係数を計算
    fir_coeff = firwin(num_taps, cutoff, fs=fs, window=window, pass_zero=False)
    return fir_coeff

def apply_fir_filter(data, fir_coeff):
    """
    FIRフィルタをデータに適用する関数

    :Parameters:
    - data: ndarray, フィルタリングする信号データ
    - fir_coeff: ndarray, FIRフィルタの係数

    :Returns:
    - filtered_data: ndarray, フィルタリング後の信号データ
    """
    filtered_data = lfilter(fir_coeff, 1.0, data)
    return filtered_data

# テスト用コード
if __name__ == "__main__":
    # サンプリング周波数
    fs = 1000  # Hz
    # 信号データを生成（サイン波 + ノイズ）
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t) + 0.5 * np.random.normal(size=t.shape)

    # ハイパスFIRフィルタを作成
    num_taps = 101  # フィルタ係数の数
    cutoff = 100  # カットオフ周波数（Hz）
    fir_coeff = create_highpass_fir_filter(num_taps, cutoff, fs)

    # フィルタを適用
    filtered_signal = apply_fir_filter(signal, fir_coeff)

    # 結果をプロット
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Original Signal")
    plt.plot(t, filtered_signal, label="Filtered Signal (Highpass)", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.title("Highpass FIR Filter Example")
    plt.show()