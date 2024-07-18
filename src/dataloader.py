import data_path
import preprocess as pp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# データセットを読み込むためのDataLoaderを定義
class DatasetLoader(Dataset):
    def __init__(self, tester, type, transform=None):
        '''
        データの初期化&前処理
        arugumets:
            tester(tester): テスターとマットレス
            type(type): 読み込むデータの種類 ("cepstrum" or "gmrf")
        '''
        self.csv_file = tester.value
        self.transform = transform
        # 前処理したrawを読み込む
        self.left, self.right, self.posture = pp.slicer("raw\\"+self.csv_file)
        self.cepstrum = pp.cmn_denoise(self.left, self.right)

    def __len__(self):
        '''
        データの大きさを返す
        '''
        return len(self.cepstrum)

    def __getitem__(self, idx):
        '''
        idxに対応するデータとラベルを返す
        arguments:
            idx (int): データのインデックス
        '''
        return self.cepstrum[idx], self.posture[idx]