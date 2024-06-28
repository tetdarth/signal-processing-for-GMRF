# サンプルデータを処理するコードは複雑なのでメンテナンスも大変
# データセットに関すコードはモジュール性を考慮し、モデルを訓練データ化r切り離すのが理想的

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root = "data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train=False,
    download=True,
    transform=ToTensor()
)
'''
labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

'''


# カスタムデータセットの作成
# 自分でカスタムしたDatasetクラスを作る際には、'__init__', `__len__`, `__getitem__` の3つの関数を必ず実装する必要がある。
# FashionMNISTの画像データを`img_dir`フォルダに、ラベルはCSVファイル`annotaions_file`として保存する。
# 各関数がどのような操作を行っているかを確認する
import os
import pandas as pd
from torchvision.io import read_image

'''
class CustomImageDataset(Dataset):
    def __init__(self, annotaions_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotaions_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label":label}
        return sample
'''


# `__init__`関数はDatasetオブジェクトがインスタンス化される際に一度だけ実行される。
# 画像、アノテーションファイル、そしてそれらに対する変換処理（transform）の初期設定を行う。

# `__len__`関数はデータセットのサンプル数を返す関数

# `__getitem__`関数は指定されたidxに対応するサンプルをデータセットから読み込んで返す関数
# indexに基づいて、画像ファイルのパスを指定し、`read_image`を使用して画像ファイルをテンソルに変換する
# 加えて、self.img_labels から対応するラベルを抜き出す
# そして、transform function を必要に応じて画像およびラベルに適用し、最終的にPythonの辞書型で画像とラベルを返す。


# DataLoaderの使用方法
# Datasetを使用することで1つのサンプルのデータとラベルを取り出せる。
# しかし、モデル訓練時にはミニバッチ単位でデータを扱いたく、また、各エポックでデータはシャッフルされてほしい
# 加えてPythonの `multiprocessing` を使用し、複数のデータの取り出しを高速化したい。
# `Data Loader` は上記に示した複雑な処理を簡単に実行してくれるAPI
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

