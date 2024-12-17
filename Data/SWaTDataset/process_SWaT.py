import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import savemat
from MTF_picture import final_MTF
from scipy import io
import time


def create_dataset(N):
    # trn = pd.read_excel('SWaT_Dataset_Normal_v1.xlsx',nrows=N+1)
    tst = pd.read_excel('SWaT_Dataset_Attack_v0.xlsx',nrows=N+1)

    # channels = tst.iloc[0].values
    # channels=channels[1:-1]

    test_data=tst.iloc[1:].values
    test_label=test_data[:,-1]
    test_data=test_data[:,1:-1]

    test_label[test_label == 'Normal'] = 0
    test_label[test_label == 'Attack'] = 1
    test_label[test_label == 'A ttack'] = 1

    return test_data, test_label


if __name__ == '__main__':

    # 设置时间戳长度
    N = 10000
    n = 5000

    # 设置开始和结束
    start = 0
    middle = start + N
    end = middle + n


    dataset='SWaT'
    data,label=create_dataset(N+n)
    print(data.shape)
    print(label.shape)


    print('unique test label', np.unique(label))
    label2=label[middle:end]
    print('unique test label', np.unique(label2))

    begin_time = time.time()

    #train
    X1=data[start:middle,:] #获取时序数据矩阵
    X2 = final_MTF(X1, N,8) #将时序数据矩阵转换成MTF矩阵
    Y = label[start:middle]
    savemat('train_'+dataset+'.mat', {'A': X1,'B':X2,'C':Y}) #存放两个矩阵于对应文件中

    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)

    #test
    X1 = data[middle:end, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, n,8)  # 将时序数据矩阵转换成MTF矩阵
    Y = label[middle:end]
    savemat('test_'+dataset+'.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中



    variables = io.loadmat('train_'+dataset+'.mat')
    print(variables['A'].shape)
    print(variables['B'].shape)
    print(variables['C'][0].shape)
    print(variables['C'])

