import numpy as np
import glob
import data_path as dpath
import preprocess as pp

print(dpath.mattresses.fl_center.value)
 # tester以外のrawを読み込む
train_cep = np.empty((0, 100))  # 100要素の配列
train_posture = np.empty(0) # 姿勢データ配列
tester_path = dpath.get_path(dpath.type.LMH, dpath.testers.H002, dpath.mattresses.fl_center)
print(str(tester_path[0]))
for p in glob.glob("raw\\LMH\\*\\" + dpath.mattresses.fl_center.value + "*", recursive=True):
    print(p)
    flag = False  # testerのパスと一致するか否かの判定
    for tpath in tester_path:
        if p == str(tpath):
            flag = True
    if flag:
        continue

    left, right, posture = pp.slicer(p)
    cep = pp.cmn_denoise(left, right)
    train_cep = np.vstack((train_cep, cep))
    train_posture = np.hstack((train_posture, posture))
print(train_cep.shape, train_posture.shape)