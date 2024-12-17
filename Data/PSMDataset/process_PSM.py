import numpy as np
import pandas as pd
from scipy.io import savemat
from MTF_picture import final_MTF
from scipy import io
import time


if __name__ == '__main__':
    dataset='PSM'
    data = pd.read_csv('test.csv').to_numpy()[:,1:]
    label = pd.read_csv('test_label.csv').to_numpy()[:,1]
    print(data.shape)

    # 设置时间戳长度
    N = 10000
    n = 1000

    # 设置开始和结束
    start = 0
    middle = start + N
    end = middle + n

    begin_time = time.time()

    #train
    X1=data[start:middle,:] #获取时序数据矩阵
    X2 = final_MTF(X1, N, 8) #将时序数据矩阵转换成MTF矩阵
    Y = label[start:middle]
    savemat('train_'+dataset+'.mat', {'A': X1,'B':X2,'C':Y}) #存放两个矩阵于对应文件中

    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)

    #test
    X1 = data[middle:end, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, n, 8)  # 将时序数据矩阵转换成MTF矩阵
    Y = label[middle:end]
    savemat('test_'+dataset+'.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中


    # variables = io.loadmat('train_'+dataset+'.mat')
    # print(variables['A'].shape)
    # print(variables['B'].shape)
    # print(variables['C'][0].shape)
