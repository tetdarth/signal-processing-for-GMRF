import torch
import numpy as np

# tensorはlistやndarrayと相互変換が可能
'''
data = [[1,2], [3,4]]
ndarr = np.array(data)
x_data = torch.tensor(ndarr)
print(x_data)

x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)
'''


# テンソルは属性変数として形状、データ型、保存されているデバイスを保持する。
tensor = torch.rand(3,4)
'''
print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
'''

# GPUが使用可能であればGPU上にテンソルを移動させる。
'''
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
'''

# numpyのようなインデックスとスライスが提供されている。
'''
tensor = torch.ones(4,4)
print('First row:',tensor[0])
print('First column',tensor[:, 0])
print('Last column',tensor[...,-1])
tensor[:,1] = 0
print(tensor)
'''

# 1要素のテンソルは.item()をすることでPythonの整数型に変換できる
'''
agg = tensor.sum() # tensorの全要素を加算
agg_item = agg.item()
print(f"agg_sum is {agg_item}")
'''

# インプレース操作(c++の`+=`のようなもの)
# メモリは節約できるが、微分を計算する際には問題となるため注意！
'''
print(tensor, "\n")
tensor.add_(5)
print(tensor)
'''



