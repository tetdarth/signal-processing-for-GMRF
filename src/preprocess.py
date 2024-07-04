import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time


# 実行時間表示
show_time = True

######################################################################
# データを切り出すパラメータ
fs = 128
frame_time = 10 # 窓サイズ (10秒)
frame = int(frame_time*fs)   # 窓サイズ
interval_time = 4    # スライス間隔  (4秒)
interval = int(interval_time*fs)

# 許容値の上限下限
tolerance_max = 4000.0
tolerance_min = 96.0

######################################################################
# 配列要素がすべて同じならTrue
def is_identical_element(data, size):
    # 0から始まる連番の二次元配列を作成
    rows = 5
    cols = size
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
    pdata = np.empty(0)            # positionの最終的な配列を格納するndarray
    i = 0   # 全体のiterate回数

    for start in range(0, wave_time-frame_time, interval_time):
        i += 1
        # waveとpositionを4秒間隔で10秒間スライス
        window = wave.iloc[start*fs : (start + frame_time)*fs]
        pos = position.iloc[start : start + frame_time]
        
        # waveとpositionをnumpy配列に変換
        rraw = window['R'].to_numpy()
        lraw = window['L'].to_numpy()
        rgain = window['R_gain'].to_numpy()
        lgain = window['L_gain'].to_numpy()
        pos = pos.to_numpy()

        # データの整合性を確認
        if not is_identical_element(rgain, frame) and not is_identical_element(lgain, frame):
            continue
        if not is_tolerance(rraw) and not is_tolerance(lraw):
            continue
        if not is_identical_element(pos, frame_time):
            continue
        if np.any(pos == 0):
            continue 

        # rawデータ
        right = (window['R'] * 2.818 ** window['R_gain']).to_numpy()
        left = (window['L'] * 2.818 ** window['L_gain']).to_numpy()
        
        # waveを2次元numpy配列として追加
        rdata = np.vstack((rdata, right))
        ldata = np.vstack((ldata, left))
        pdata = np.append(pdata, pos[0])
            
    if show_time:
        t2 = time.time()    # 時間計測end
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time:.3}[s]")
    
    print(f"data[{len(pdata)} / {i}]")
    # print(pdata)
    
    '''
    count = {1:0, 2:0, 3:0, 4:0}
    for x in pdata:
        count[x] += 1
    print(count[4])
    '''

create_dataset("raw\\LMH\\H002\\H002_fl_center\\")
#create_dataset("raw\\LMH\\H002\\H002_ka_left\\")
#create_dataset("raw\\LMH\\H002\\H002_ka_right\\")
#create_dataset("raw\\LMH\\H002\\H002_ka_center\\")
#create_dataset("raw\\LMH\\H002\\H002_st_center\\")
