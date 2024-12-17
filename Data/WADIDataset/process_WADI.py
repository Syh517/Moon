import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from scipy.io import savemat
from MTF_picture import final_MTF
from scipy import io
import time


def create_dataset():
    #19
    train_new = pd.read_csv('WADI_14days_new.csv')
    test_new = pd.read_csv('WADI_attackdataLABLE.csv', skiprows=1)

    #17
    # test_new = pd.read_csv('WADI_attackdata.csv')
    # train_new = pd.read_csv('WADI_14days.csv', skiprows=4)

    ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
    train_new = train_new.drop(ncolumns, axis=1)
    test_new = test_new.drop(ncolumns, axis=1)
    train_new = train_new.dropna(axis=0, how='all')
    test_new = test_new.dropna(axis=0, how='all')
    test_new = test_new.iloc[:, 3:]
    train_new = train_new.iloc[:, 3:]

    test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'}, inplace=True)
    test_new.loc[test_new['label'] == 1, 'label'] = 0
    test_new.loc[test_new['label'] == -1, 'label'] = 1
    wadi_labels = test_new['label']
    test_new = test_new.iloc[:, :-1]

    from sklearn.preprocessing import MinMaxScaler

    # 最大最小值归一化
    scaler = MinMaxScaler()  # 实例化
    wadi_train = scaler.fit_transform(train_new)
    wadi_test = scaler.fit_transform(test_new)

    return wadi_test, wadi_labels.values



if __name__ == '__main__':
    dataset = 'WADI'
    data, label = create_dataset()
    print(data.shape)
    print(label.shape)
    # print(data)
    print(label)

    # 设置时间戳长度
    N = 6000
    n = 3000

    # 设置开始和结束
    start = 0
    middle = start + N
    end = middle + n

    label1 = label[start:middle]
    print('unique test label', np.unique(label1))
    label2 = label[middle:end]
    print('unique test label', np.unique(label2))

    begin_time = time.time()

    # train
    X1 = data[start:middle, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, N, 9)  # 将时序数据矩阵转换成MTF矩阵
    Y = label[start:middle]
    savemat('train_' + dataset + '.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中


    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)

    # test
    X1 = data[middle:end, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, n, 9)  # 将时序数据矩阵转换成MTF矩阵
    Y = label[middle:end]
    savemat('test_' + dataset + '.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中


    variables = io.loadmat('train_'+dataset+'.mat')
    print(variables['A'].shape)
    print(variables['B'].shape)
    print(variables['C'][0].shape)

