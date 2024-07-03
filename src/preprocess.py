import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def create_dataset():
    # pandasでcsvを読み込み
    wave = pd.read_csv("data\\raw\\LMH\\H002\\H002_fl_center\\wave.csv", names=["R", "L", "R_gain", "L_gain"])

    # データを切り出すパラメータ
    fs = 128
    frame_time = 10 # 窓サイズ (10秒)
    frame = int(frame_time*fs)   # 窓サイズ
    interval_time = 4    # スライス間隔  (4秒)
    interval = int(interval_time*fs)

    # waveの長さ [s]
    wave_time = int(len(wave)/fs)
    rdata = np.empty((0, frame))   # rightの最終的な配列を格納するndarray
    ldata = np.empty((0, frame))   # leftの最終的な配列を格納するndarray

    # 1から始まる連番の二次元配列を作成
    rows = 5
    cols = frame
    template = np.tile(np.arange(1, rows + 1), (cols, 1)).T

    for start in range(0, wave_time-frame_time, interval_time):
        # start = 0,4,8,12,...
        # waveを4秒間隔で10秒間スライス
        window = wave.iloc[start*fs : (start + frame_time)*fs]
        right = window['R'] * 2.818 ** window['R_gain']
        left = window['L'] * 2.818 ** window['L_gain']
        
        # dataをnumpy配列に変換
        right = right.to_numpy()
        left = left.to_numpy()
        rgain = window['R_gain'].to_numpy()
        lgain = window['L_gain'].to_numpy()

        # データの整合性を確認
        is_integrity = False
        for num_template in template:
            if np.array_equal(rgain, num_template) & np.array_equal(lgain, num_template):
                is_integrity = True

        # dataを2次元numpy配列として追加
        if is_integrity:
            rdata = np.vstack((rdata, right))
            ldata = np.vstack((ldata, left))
        

    print(len(rdata))
    print(rdata)

create_dataset()