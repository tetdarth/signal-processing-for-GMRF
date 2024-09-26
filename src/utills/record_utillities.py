import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Arial'

def save_confusion_matrix(conf_mat):
    '''
    args:
        conf_mat : ndarray
    '''
    pd.options.display.precision = 4
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mat, cmap = 'Blues', annot=True)
    plt.savefig("../images/confution_matrix.jpg")
    plt.show()
    
