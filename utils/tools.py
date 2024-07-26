import numpy as np
import torch
from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def caculate(_x_data):
    _x_data = _x_data[:,:,(-1,-2,-3)].cpu()
    n = _x_data.shape[-1]
    data = pd.DataFrame(_x_data.numpy().reshape(-1, n))
    cor = data.corr(method='pearson')
    cor = cor.abs()
    row_sums = cor.sum(axis=1)-1
    a1 = row_sums[0]/np.sum(row_sums)
    a2 = row_sums[1]/np.sum(row_sums)
    a3 = row_sums[2]/np.sum(row_sums)
    return a1,a2,a3


def ShowHeatMap(DataFrame, method, title):
    if method == 'MIC':
        colormap = plt.cm.RdBu
        ylabels = DataFrame.columns.values.tolist()
        f, ax = plt.subplots(figsize=(14, 14))
        ax.set_title('GRA HeatMap ' + title,fontname="SimHei")
        sns.heatmap(DataFrame.astype(float),
                    cmap=colormap,
                    ax=ax,
                    annot=True,
                    yticklabels=ylabels,
                    xticklabels=ylabels)
        plt.show()
    elif method == 'pearson':
        plt.figure(figsize=(19, 16))
        cor = DataFrame.corr(method='pearson')
        # labels中填充要计算相关性的列
        lables = []
        rc = {'font.sans-serif': 'SimHei',
              'axes.unicode_minus': False}
        sns.set(font_scale=1.6, rc=rc)  # 设置字体大小
        sns.heatmap(cor,
                    annot=True,  # 显示相关系数的数据
                    center=0.35,  # 居中
                    fmt='.2f',  # 只显示两位小数
                    linewidth=0.5,  # 设置每个单元格的距离
                    linecolor='blue',  # 设置间距线的颜色
                    vmin=-1, vmax=1,  # 设置数值最小值和最大值
                    xticklabels=lables, yticklabels=lables,  # 显示x轴和y轴
                    square=True,  # 每个方格都是正方形
                    cbar=True,  # 绘制颜色条
                    cmap='coolwarm_r',  # 设置热力图颜色
                    )
        plt.xticks(rotation=45)
        plt.subplots_adjust(right=0.9999)
        plt.subplots_adjust(top=0.95)
        plt.show()  # 显示图片


def MIC(X, Y):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(X, Y)
    mic = mine.mic()
    return mic
