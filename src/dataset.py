import numpy as np
import pandas as pd
import tester
import preprocess as pp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# データセットを読み込むためのDataLoaderを定義
class DatasetLoader(Dataset):
    def __init__(self, tester, transform=None):
        '''
        データの初期化&前処理
        arugumets:
            tester(tester): csvファイルの属性
        '''
        self.csv_file = tester.value
        self.transform = transform
        # 前処理したrawを読み込む
        self.data, self.posture = pp.preprocess("raw\\"+self.csv_file)

    def __len__(self):
        '''
        データの大きさを返す'''
        return len(self.data[0])

    def __getitem__(self, idx):
        '''
        idxに対応するデータとラベルを返す
        arguments:
            idx (int): データのインデックス'''
        return self.data[idx], self.posture[idx]

tester = tester.H002.fl_center.value
data, posture = pp.preprocess("raw\\"+tester)
print(len(data[0]))