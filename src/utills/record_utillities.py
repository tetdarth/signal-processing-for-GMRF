import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import re
from PIL import Image
from IPython.display import Image as IPImage, display

plt.rcParams['font.family'] = 'Arial'

# 混同行列の生成
def save_confusion_matrix(conf_mat):
    '''
    args:
        conf_mat : ndarray
    '''
    pd.options.display.precision = 4
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mat, cmap = 'Blues', annot=True, fmt='.0f')
    plt.savefig("../images/confution_matrix.jpg")
    plt.show()
    
# 箱ひげ図の生成
def boxplot(data, labels, save=True, title='', attribute=''):
    '''
    args:
        data : ndarray
        save : bool
        title : String
    '''
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    plt.title(title)
    plt.grid()
    if save:
        path = "../images/"+title
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path+'/'+attribute+"_boxplot.png")
    plt.show()

# csvの情報から箱ひげ図の生成
def csv_to_boxplot(type, tester, attr='', save=True):
    '''
    args:
        type : String
        tester : String
        attr : String (HAR, SEHAR, DNN)
        save : bool
    '''
    csv_path = glob.glob(f"../tmp/{type}_{tester}/*_{attr}.csv")
    save_path = "../images/"+f"{type}_{tester}"
    if save:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    accuracy = []
    error = []
    position = []

    for p in csv_path:
        # DataFrameの読み込み
        data = pd.read_csv(p)
        acc = data['acc'].to_numpy()
        err = data['err'].to_numpy()
        pos = re.search(r'(ka|st|fl|fls|fld|flf)_(center|left|right)', p)
        # リストに追加
        accuracy.append(acc[20:])
        error.append(err[20:])
        position.append(pos.group())

    # print(accuracy)

    # accuracy
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(accuracy, patch_artist=True)
    ax1.set_xticklabels(position)
    ax1.set_ylim(0, 100)
    plt.title(f"{attr}_acc")
    plt.grid()
    if save:
        plt.savefig(save_path + f"/{attr}_acc.png")

    # error
    fig, ax2 = plt.subplots()
    bp = ax2.boxplot(error, patch_artist=True)
    ax2.set_xticklabels(position)
    plt.title(f"{attr}_err")
    plt.grid()
    if save:
        plt.savefig(save_path + f"/{attr}_err.png")
    
    plt.close()


def boxplot_plot() -> None:
    type = ['LMH']
    tester = ['H002', 'H003', 'L001', 'L003', 'M001', 'M002', 'M003', 'M004']
    attr = ['HAR_norm=backward']

    for i in type:
        for t in tester:
            for a in attr:
                csv_to_boxplot(i,t,a)


# GIFアニメーションの生成
def make_gif(target_dir, gif_path) -> None:
    # 画像のpathを取得
    image_paths = sorted(glob.glob(f"{target_dir}/*.png"), key=os.path.getmtime)
    # 画像を開く（表示しない）
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.copy())  # copyして保持、withブロックで自動close

    # GIFを作成
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # フレームの表示時間 (ミリ秒)
        loop=0  # 0は無限ループ
    )

    # pngを削除
    for path in image_paths:
        os.remove(path)

    # GIFの表示
    display(IPImage(filename=gif_path))
