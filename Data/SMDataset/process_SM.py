import numpy as np
import pandas as pd
from scipy.io import savemat
from MTF_picture import final_MTF
from scipy import io
import time

if __name__ == '__main__':
    dataset='SM'

    train_data = np.loadtxt('train.txt', dtype=np.float32, delimiter=',')
    train_label = np.loadtxt('train_label.txt', dtype=np.float32, delimiter=',')

    test_data = np.loadtxt('test.txt', dtype=np.float32, delimiter=',')
    test_label = np.loadtxt('test_label.txt', dtype=np.float32, delimiter=',')


    # 设置时间戳长度
    N = 10000
    n = 10000

    start = 0

    begin_time = time.time()

    #train
    X1=train_data[start:N,:] #获取时序数据矩阵
    X2 = final_MTF(X1, N, 9) #将时序数据矩阵转换成MTF矩阵
    Y = train_label[start:N]
    savemat('train_'+dataset+'.mat', {'A': X1,'B':X2,'C':Y}) #存放两个矩阵于对应文件中

    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)

    #test
    X1 = test_data[start:n, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, n, 9)  # 将时序数据矩阵转换成MTF矩阵
    Y = test_label[start:n]
    savemat('test_'+dataset+'.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中



    # variables = io.loadmat('train_'+dataset+'.mat')
    # print(variables['A'].shape)
    # print(variables['B'].shape)
    # print(variables['C'][0].shape)
