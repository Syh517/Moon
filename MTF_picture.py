import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time

from matplotlib import gridspec
from numba import njit, prange
from pyts.image import MarkovTransitionField
from pyts.preprocessing.discretizer import KBinsDiscretizer

import tsia.plot
import tsia.markov
import tsia.network_graph
from scipy.io import savemat
from scipy import io


#绘制单变量的时序图
def shows_ts(ts,i):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(ts[:,i], linewidth=3)
    plt.title('Time Series Data-'+str(i), fontsize=18)
    plt.tight_layout()
    plt.ylim(0,5)
    plt.show()

#绘制单变量的马尔可夫转移场热力图
def shows_mtf(mtf,i):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=mtf, ax=ax, title='Markov Transition Field '+str(i), reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.show()

#绘制单变量的聚集马尔可夫转移场热力图
def shows_amtf(amtf,i):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=amtf, ax=ax, title='Aggregated Markov Transition Field '+str(i), reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.show()

#绘制双变量的马尔可夫转移场热力图
def showd_mtf(mtf,i,j):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=mtf, ax=ax, title='Markov Transition Field '+str(i)+'-'+str(j), reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.show()

#绘制双变量的聚集马尔可夫转移场热力图
def showd_amtf(amtf,i,j):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=amtf, ax=ax, title='Aggregated Markov Transition Field '+str(i)+'-'+str(j), reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.show()


def show_final_mtf(mtf):
    fig = plt.figure(figsize=(2, 18))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=mtf, ax=ax, title='Markov Transition Field ', reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.show()


#求单变量的状态转移场矩阵
def single_markov_transition_field(X_binned, X_mtm, n_timestamps):
    X_mtf = np.zeros((n_timestamps, 1))

    X_mtf[0, 0]=X_mtm[X_binned[0], X_binned[0]]
    # We loop through each timestamp twice to build a N x N matrix:
    for i in range(n_timestamps-1):
       X_mtf[i, 0] = X_mtm[X_binned[i], X_binned[i+1]]

    return X_mtf

#求双变量的状态转移场矩阵
def multi_markov_transition_field(X_binned, Y_binned, XY_mtm, n_timestamps):
    XY_mtf = np.zeros((n_timestamps, 1))

    XY_mtf[0, 0] = XY_mtm[X_binned[0], Y_binned[0]]
    # We loop through each timestamp twice to build a N x N matrix:
    for i in range(n_timestamps-1):
        XY_mtf[i, 0] = XY_mtm[X_binned[i], Y_binned[i+1]]

    return XY_mtf

#求单变量的马尔可夫转移场矩阵
def single_MTF(V_binned,n_timestamps,n_bins):
    #计算状态转移矩阵
    V_mtm = tsia.markov.markov_transition_matrix(V_binned)
    V_mtm = tsia.markov.markov_transition_probabilities(V_mtm)

    #计算马尔可夫转移场矩阵
    V_mtf=single_markov_transition_field(V_binned, V_mtm, n_timestamps)

    return V_mtf

#求双变量的马尔可夫转移场矩阵
def multi_MTF(X_binned, Y_binned, n_timestamps, n_bins):
    # 计算状态转移矩阵
    XY_mtm = np.zeros((n_bins, n_bins))
    #遍历所有时序，找到状态转移频次
    for t in prange(n_timestamps - 1):
        XY_mtm[X_binned[t], Y_binned[t + 1]] += 1
    #将状态转移频次归一化，转换为概率
    XY_mtm = tsia.markov.markov_transition_probabilities(XY_mtm)

    # 计算马尔可夫转移场矩阵
    XY_mtf = multi_markov_transition_field(X_binned, Y_binned, XY_mtm, n_timestamps)

    return XY_mtf


def final_MTF(X,n_timestamps,n_bins=8): #X:(n_timestamps,38)
    Vn=X.shape[1]
    p=0.9 #时序数据转MTF矩阵的矩阵权重参数

    #对所有数据进行分桶
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy='quantile')
    Binned=[]
    for i in range(Vn):
        V = X[:, i].reshape(1, -1)
        V_binned = discretizer.fit_transform(V)[0]
        Binned.append(V_binned)

    #计算MTF
    list_mtf=[]
    for i in range(Vn):
        self_MTF=single_MTF(Binned[i], n_timestamps, n_bins)
        rest_MTF = np.zeros((n_timestamps, 1))
        for j in range(Vn):
            if j!=i:
                rest_MTF = rest_MTF + multi_MTF(Binned[i],Binned[j], n_timestamps, n_bins)
        # rest_MTF=rest_MTF/(Vn-1)
        # f_MTF = self_MTF * p + rest_MTF * (1 - p)

        f_MTF=(self_MTF*p+rest_MTF*(1-p))/Vn

        # f_MTF=(self_MTF+rest_MTF)/Vn

        # f_MTF=rest_MTF

        list_mtf.append(f_MTF)

    F_MTF=np.concatenate(list_mtf,axis=1) #拼接
    return F_MTF



# 设置文件路径
data_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save2/'
dataset_name_list = [
    '1-1',
    # '1-2',
    # '1-3',
    # '1-4',
    # '1-5',
    # '1-6',
    # '1-7',
    # '1-8',
    # '2-1',
    # '2-2',
    # '2-3',
    # '2-4',
    # '2-5',
    # '2-6',
    # '2-7',
    # '2-8',
    # '2-9',
    # '3-1',
    # '3-2',
    # '3-3',
    # '3-4',
    # '3-5',
    # '3-6',
    # '3-7',
    # '3-8',
    # '3-9',
    # '3-10',
    # '3-11',
]


if __name__ == '__main__':

    N=10000
    #设置时间戳长度
    n_timestamps =N


    #设置开始和结束
    start=0
    end=start+n_timestamps

    #设置存放矩阵的文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # 将所有数据转换成2类矩阵并存放在.mat文件中
    # for dataset_name in dataset_name_list:
    #     X = np.loadtxt(data_path + 'test/machine-'+dataset_name+'.txt', dtype=np.float32, delimiter=',') #获取数据集数据
    #     X1=X[start:end,:] #获取时序数据矩阵
    #     X2 = final_MTF(X1, n_timestamps) #将时序数据矩阵转换成MTF矩阵
    #     Y = np.loadtxt(data_path + 'test_label/machine-'+dataset_name+'.txt')
    #     Y=Y[start:end]
    #     savemat(save_path + 'machine-' + dataset_name + '.mat', {'A': X1,'B':X2,'C':Y}) #存放两个矩阵于对应文件中
    #     print(save_path + 'machine-'+dataset_name+'.mat')
    #
    #
    # variables = io.loadmat(save_path + 'machine-'+'3-7'+'.mat')
    # print(variables)
    # print(variables['A'].shape)
    # print(variables['B'].shape)
    # print(variables['C'][0].shape)


    begin_time = time.time()

    for dataset_name in dataset_name_list:
        X = np.loadtxt(data_path + 'test/machine-' + dataset_name + '.txt', dtype=np.float32, delimiter=',')  # 获取数据集数据
        X1 = X[start:end, :]  # 获取时序数据矩阵
        X2 = final_MTF(X1, n_timestamps)  # 将时序数据矩阵转换成MTF矩阵

    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)
