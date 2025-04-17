import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import utills.datapath as dpath
import utills.preprocess as pp
import torch.optim as optim
import torch.nn as nn
import model.har as har
import model.sehar as sehar
import model.dnn as dnn
from torcheval.metrics import MulticlassConfusionMatrix
import random
import os
import warnings
import pandas as pd
from utills import record_utillities as ru

def seed_everything(seed=0):
    random.seed(seed)  # Python標準のrandomモジュールのシードを設定
    os.environ['PYTHONHASHSEED'] = str(seed)  # ハッシュ生成のためのシードを環境変数に設定
    np.random.seed(seed)  # NumPyの乱数生成器のシードを設定
    torch.manual_seed(seed)  # PyTorchの乱数生成器のシードをCPU用に設定
    torch.cuda.manual_seed(seed)  # PyTorchの乱数生成器のシードをGPU用に設定
    torch.backends.cudnn.deterministic = True  # PyTorchの畳み込み演算の再現性を確保

seed_everything()  # 上述のシード設定関数を呼び出し

warnings.simplefilter('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class dataset(Dataset):
    def __init__(self, paths, concat=True, transforms=None):
        self.transforms = transforms
        self.concat = concat

        # データを格納する配列を確保
        if concat:
            self.train_cepstrum = np.empty((0, 100))  # 100要素の配列
        else:
            self.train_cepstrum = []
        self.train_posture = np.empty(0)  # 姿勢データ配列

        # データ読み込みと前処理
        for p in paths:
            left, right, posture = pp.slicer(p)
            cepstrum = pp.cmn_denoise(left, right, concat=concat)
            for cep in cepstrum:
                if concat:
                    self.train_cepstrum = np.vstack((self.train_cepstrum, cep)) if self.train_cepstrum.size else cep
                else:
                    self.train_cepstrum.append(cep)
            self.train_posture = np.append(self.train_posture, posture) if self.train_posture.size else posture

    def __len__(self):
        return len(self.train_posture)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        if self.concat:
            cepstrum = torch.tensor(self.train_cepstrum[idx].reshape(1,-1), dtype=torch.float32)
        else:
            cepstrum = torch.tensor(self.train_cepstrum[idx], dtype=torch.float32)
        posture = torch.tensor(self.train_posture[idx]-1, dtype=torch.long)
        if self.transforms is not None:
            cepstrum = self.transforms(cepstrum)
            posture = self.transforms(posture)
        return cepstrum, posture
    
validation = 'M001'
identities = dpath.LMH.all()
identities = dpath.filter(identities, validation)
print(identities)
# バッチサイズ
batch_size = 128

test_accuracies = []
test_errors = []

# モデルの種類 (HAR, SEHAR, DNN)
attr = 'HAR'
# データの形
concat = True if attr=='DNN' else False

# 被験者
for identity in identities:
    type, tester, mattress, _ = dpath.getattributes(identity)
    train_paths, val_paths, test_path = dpath.get_paths(identity, validatioin='M001')

    print("--- train ---")
    train = dataset(train_paths, concat=concat)
    print("--- validation ---")
    val = dataset(val_paths, concat=concat)
    print("--- test ---")
    test = dataset(test_path, concat=concat)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True
    )

    test_loader = DataLoader(
        test,
        shuffle=False,
    )

    # 学習
    # モデルのインスタンス化
    num_channel = 1 if concat else 2

    if attr == 'SEHAR':
        net = sehar.SE_HAR(num_classes=4, num_channel=num_channel).to(device)
    elif attr == 'HAR':
        net = har.HAR_resnet18(num_classes=4, num_channel=num_channel).to(device)
    elif attr == 'DNN':
        net = dnn.dnn(n_input=100, n_output=4).to(device)
    else:
        assert()

    # 誤差関数を交差エントロピーで計算
    criterion = nn.CrossEntropyLoss()

    # 最適化アルゴリズム
    lr = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)

    confusion_mat = MulticlassConfusionMatrix(
            num_classes = 4
    )

    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_error, val_error, test_error = [], [], []

    # 学習
    n_epoch = 50
    for epoch in range(n_epoch):
        # 精度と損失の初期化
        train_acc, train_loss = 0, 0
        val_acc, val_loss = 0, 0
        n_train, n_val = 0, 0
        test_loss, test_acc = 0, 0
        n_test = 0

        # 学習
        for train_input, train_label in train_loader:
            n_train += len(train_label)

            # 入力と正解ラベルをGPU上に移動
            input = train_input.to(device)
            label = train_label.to(device)
            # print(f'input : {input.shape}, label : {label.shape}')

            # モデルを学習モードに変更
            net.train()

            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output, 1)[1]

            train_loss += loss.item()
            train_acc += (predicted == label).sum().item()

        # 検証
        for val_input, val_label in val_loader:
            n_val += len(val_label)

            val_input = val_input.to(device)
            val_label = val_label.to(device)

            # モデルを推論モードに変更
            net.eval()

            with torch.no_grad():
                val_output = net(val_input)
            loss = criterion(val_output, val_label)

            val_predicted = torch.max(val_output, 1)[1]
            
            val_loss += loss.item()
            val_acc += (val_predicted == val_label).sum().item()
            if epoch+1 == n_epoch:
                confusion_mat.update(val_predicted, val_label)

        # テスト
        for test_input, test_label in test_loader:
            n_test += len(test_label)

            test_input = test_input.to(device)
            test_label = test_label.to(device)

            # モデルを推論モードに変更
            net.eval()

            with torch.no_grad():
                test_output = net(test_input)
            loss = criterion(test_output, test_label)

            test_predicted = torch.max(test_output, 1)[1]

            test_loss += loss.item()
            test_acc += (test_predicted == test_label).sum().item()
            
        # 精度を確率に変換
        test_acc /= n_test
        test_loss = test_loss / n_test

        print(f"loss : {test_loss:.5f}, acc : {test_acc:.5f}")

        # 精度を確率に変換
        train_acc /= n_train
        val_acc /= n_val
        train_loss = train_loss * batch_size / n_train
        val_loss = val_loss * batch_size / n_val

        train_accuracy.append(train_acc*100)
        val_accuracy.append(val_acc*100)
        test_accuracy.append(test_acc*100)
        train_error.append(train_loss)
        val_error.append(val_loss)
        test_error.append(test_loss)

        if epoch+1 == n_epoch:
            confusion_mat.compute()

        if not epoch%1:
            print(f"Epoch[{epoch+1}/{n_epoch}] | train_loss: {train_loss:.5f} | train_acc: {train_acc:.5f} | val_loss: {val_loss:.5f} | val_acc: {val_acc:.5f}")

    test_accuracies.append(test_accuracy)
    test_errors.append(test_error)
    _, _, mattress, _ = dpath.getattributes(identity, include_position=True)

    # csvの保存
    df = pd.DataFrame({'acc' : test_accuracy, 'err' : test_error}, index=np.arange(n_epoch))
    path = f"../tmp/{type}_{tester}"
    if not os.path.isdir(path):
        os.mkdir(path)
    df.to_csv(path+f"/{mattress}_{attr}_HAR2.csv")

# 箱ひげ図のプロット
ru.boxplot()
