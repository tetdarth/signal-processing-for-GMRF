import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
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
data_path = tester.H002.fl_center.value

# データを切り出すパラメータ
fs = 128        # サンプリング周波数
frame_time = 10     # 窓サイズ (10秒)
interval_time = 4    # スライス間隔  (4秒)
frame = frame_time*fs   # 窓幅
interval = interval_time*fs    # スライス幅

# 不整合データのパラメータ
tolerance_max = 4000.0
tolerance_min = 96.0

'''################ utillity #####################'''
# 配列要素がすべて同じならTrue
def is_identical_element(data):
    # 1から始まる連番の二次元配列を作成
    rows = 5
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
def wave_plot(wave):
    sample = np.arange(len(wave))
    plt.plot(sample, wave)
    plt.show()

'''################# preprocess ####################'''
def preprocess(dir):
    # pandasでcsvを読み込み
    wave = pd.read_csv(dir+"\\wave.csv", names=["R", "L", "R_gain", "L_gain"])
    position = pd.read_csv(dir+"\\position.csv")

    # waveの長さ [s]
    wave_time = int(len(wave)/fs)
    rdata = np.empty((0, frame))   # rightの最終的な配列を格納するndarray
    ldata = np.empty((0, frame))   # leftの最終的な配列を格納するndarray
    pdata = np.empty(0)
    i = 0

    # 前処理
    for start in range(0, wave_time-frame_time, interval_time):
        i += 1
        # waveとpositionを4秒間隔で10秒間スライス
        window = wave.iloc[start*fs : (start + frame_time)*fs]
        pos = position.iloc[start : start + frame_time]

        # waveとをnumpy配列に変換
        rraw = window['R'].to_numpy()
        lraw = window['L'].to_numpy()
        rgain = window['R_gain'].to_numpy()
        lgain = window['L_gain'].to_numpy()
        pos = pos.to_numpy()

        # データの整合性を確認
        if not is_identical_element(rgain) and not is_identical_element(lgain):
            continue
        if not is_tolerance(rraw) and not is_tolerance(lraw):
            continue
        if not is_identical_element(pos):
            continue
        if np.any(pos == 0):
            continue

        # 波形の復元
        right = (window['R'] * 2.818 ** window['R_gain']).to_numpy()
        left = (window['L'] * 2.818 ** window['L_gain']).to_numpy()

        # 波形の正規化
        right, left = c.normalize(right, left)

        # dataを2次元numpy配列として追加
        rdata = np.vstack((rdata, right))
        ldata = np.vstack((ldata, left))
        pdata = np.append(pdata, pos[0])

    print(f"data[{len(rdata)} / {i}]")
    return rdata, ldata, pdata

def create_dataset(dir):
    if show_time:
        t1 = time.time()   # 時間計測start

    # ディレクトリ内のcsvファイルを削除
    for p in glob.glob(".\\datasets\\" + dir +"\\***", recursive=True):
        if os.path.isdir(p):
            shutil.rmtree(p)

    # rawデータのスライス
    right, left, position = preprocess("raw\\"+dir)

    # データをcsvに書き出し
    for i in range(len(right)):
        # ndarrayをDataFrameに変換
        rdata = pd.DataFrame(right[i])
        ldata = pd.DataFrame(left[i])
        data = pd.concat([rdata, ldata], axis=1).astype(np.float64)
        pdata = position[i]

        # DataFrameをcsvに書き出し
        path = ".\\datasets\\" + dir +"\\" + str(int(pdata))
        if not os.path.exists(path):
            os.makedirs(path)
        files = sum(os.path.isfile(os.path.join(path, name)) for name in os.listdir(path))
        data.to_csv(path + "\\" + str(files+1) + ".csv")

    if show_time:
        t2 = time.time()    # 時間計測end
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time:.3}[s]")


create_dataset(data_path)