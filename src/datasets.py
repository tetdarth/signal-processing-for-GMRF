import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time


# 実行時間表示
show_time = True

#####################################
# データを切り出すパラメータ
fs = 128
frame_time = 10 # 窓サイズ (10秒)
frame = int(frame_time*fs)   # 窓サイズ
interval_time = 4    # スライス間隔  (4秒)
interval = int(interval_time*fs)

# 不整合データのパラメータ
integre_max = 14000.0
integre_min = 96.0

#####################################
# 1から始まる連番の二次元配列を作成
rows = 5
cols = frame
template = np.tile(np.arange(0, rows), (cols, 1)).T

# 配列要素がすべて同じならTrue
def is_identical_element(data):
    for num_template in template:
        if np.array_equiv(data, num_template):
            return True
    return False

# 配列が正常値内に存在していればTrue
def is_tolerance(data):
    if not integre_min <= np.min(data) or not np.max(data) <= integre_max:
        return False
    return True

#####################################
def create_dataset(dir):
    if show_time:
        t1 = time.time()   # 時間計測start
    
    # pandasでcsvを読み込み
    wave = pd.read_csv(dir+"wave.csv", names=["R", "L", "R_gain", "L_gain"])
    position = pd.read_csv(dir+"position.csv")

    # waveの長さ [s]
    wave_time = int(len(wave)/fs)
    rdata = np.empty((0, frame))   # rightの最終的な配列を格納するndarray
    ldata = np.empty((0, frame))   # leftの最終的な配列を格納するndarray
    i = 0

    for start in range(0, wave_time-frame_time, interval_time):
        i += 1
        # waveを4秒間隔で10秒間スライス
        window = wave.iloc[start*fs : (start + frame_time)*fs]
        right = window['R'] * 2.818 ** window['R_gain']
        left = window['L'] * 2.818 ** window['L_gain']
        # potisionを4秒感覚で10秒間スライス
        pos = position.iloc[start : start + frame_time]
        
        # waveをnumpy配列に変換
        right = right.to_numpy()
        left = left.to_numpy()
        rgain = window['R_gain'].to_numpy()
        lgain = window['L_gain'].to_numpy()
        
        # positionをnumpy配列に変換
        pos = pos.to_numpy()

        # データの整合性を確認
        is_integrity = True
        if not is_identical_element(rgain) and not is_identical_element(lgain):
            is_integrity = False
        if not is_tolerance(right) and not is_tolerance(left):
            is_integrity = False
        if not is_identical_element(pos):
            is_integrity = False
        if np.any(pos == 0):
            is_integrity = False

        # dataを2次元numpy配列として追加
        if is_integrity:
            rdata = np.vstack((rdata, right))
            ldata = np.vstack((ldata, left))
            
    if show_time:
        t2 = time.time()    # 時間計測end
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time:.3}[s]")
    
    print(f"data[{len(rdata)} / {i}]")

create_dataset("raw\\LMH\\H002\\H002_fl_center\\")
create_dataset("raw\\LMH\\H002\\H002_ka_center\\")
create_dataset("raw\\LMH\\H002\\H002_ka_left\\")
create_dataset("raw\\LMH\\H002\\H002_ka_right\\")
create_dataset("raw\\LMH\\H002\\H002_st_center\\")
