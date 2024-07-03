import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# mnistは28*28 = 784ユニット
# 中間層は128ユニット
# 出力層が10ユニット

# GPUが使用可能か確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# MINISTのデータセットをダウンロード
data_root = './data'

# 前処理
transform = transforms.Compose([
    transforms.ToTensor(),  # データをテンソルに変換
    transforms.Normalize(0.5, 0.5),     # [-1, 1] に正規化
    transforms.Lambda(lambda x : x.view(-1))    # 3階テンソルを1階テンソルに変換
])

# 訓練データ
train_set = datasets.MNIST(
    root = data_root,
    train = True,
    download = True,
    transform = transform
)

# テストデータ
test_set = datasets.MNIST(
    root = data_root,
    train = False,
    download = True,
    transform = transform
)

# テンソルになっているかを確認
image, label = train_set[0]
print('image type:', type(image))
print('image shape:', image.shape)

# 各画像のデータがどのような値をとるか(最小値、最大値)
print('min: ', image.data.min())
print('max: ', image.data.max())

# 訓練データの長さを確認
print('train_size : ',len(train_set))
print('test_size : ',len(test_set))

# データを一つ可視化してみる
'''
image, label = train_set[0]
plt.figure(figsize=(2,2))
plt.title(f'{label}')
plt.imshow(image[0], cmap='gray_r')
plt.show()
'''

batch_size = 500
train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle=True,
    pin_memory=True
)
test_loader = DataLoader(
    test_set,
    batch_size = batch_size,
    shuffle=True,
    pin_memory=True
)
print(len(train_loader))

# 各層のノードの個数
n_input = 28*28
n_output = 10
n_hidden = 128

# pytorchはオブジェクト指向をメインにNNを構築
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        # 各層を全結合(Linear)
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.Linear(n_hidden, 64)
        self.layer3 = nn.Linear(64, n_output)
        self.relu = nn.ReLU(inplace=True)
        
    # 各層を結合
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
# モデルのインスタンス化
# pytorchの場合はto(device)でモデルをGPU上に移動する必要がある
net = Net(n_input, n_output, n_hidden).to(device)

# 誤差関数を交差エントロピーで計算（分類なので）
# 誤差関数もクラスで定義されている。
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズムを定義
# 今回は確率的勾配法(SGD)を用いて計算
lr = 0.01
optimizer = optim.Adam(net.parameters(), lr = lr)

# 評価結果を格納するベクタ
history = np.zeros((0, 5))

############## 学習 ################
num_epoch = 100
for epoch in range(num_epoch):
    # 精度と損失の初期化
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0
    
    # 学習
    for input, label in train_loader:
        n_train += len(label)
        
        # 入力と正解ラベルをGPU上に移動
        input = input.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()   # 勾配の初期化
        output = net(input)     # 順伝播
        loss = criterion(output, label)     # 損失の計算(出力, 正解ラベル)
        loss.backward()     # 誤差逆伝播
        optimizer.step()    # 勾配の更新
        
        predicted = torch.max(output, 1)[1] # 出力層の最も確率の高いノードを出力とする
        
        train_loss += loss.item()
        train_acc += (predicted == label).sum().item()      # 500回の内何回正解したか
        
        
    # 検証
    for test_input, test_label in test_loader:
        n_test += len(test_label)
        
        test_input = test_input.to(device)
        test_label = test_label.to(device)
        
        test_output = net(test_input)
        test_loss = criterion(test_output, test_label)
        
        test_predicted = torch.max(test_output, 1)[1]
        
        val_loss += test_loss.item()
        val_acc += (test_predicted == test_label).sum().item()
        
    # 精度を確率に変換
    train_acc = train_acc / n_train 
    val_acc = val_acc / n_test
    # 損失を計算
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
        
    print(f"Epoch[{epoch+1}/{num_epoch}], | loss: {train_loss:.5f} | acc: {train_acc:.5f} | val_loss: {val_loss:.5f} | val_acc: {val_acc:.5f}")
    items = np.array([epoch+1, train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history, items))
    