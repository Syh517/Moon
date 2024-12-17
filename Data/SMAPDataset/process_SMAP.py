import numpy as np
import pandas as pd
from scipy.io import savemat
from MTF_picture import final_MTF
from scipy import io
import time

pth = './'

labeled_anomalies = pd.read_csv('labeled_anomalies.csv')

data_dims = {'SMAP': 25, 'MSL': 55}

def create_dataset(smap_or_msl):
    print(f'Creating dataset for {smap_or_msl}')
    train_data = []
    test_data = []
    test_label = []
    total_anomaly_points = 0
    for i in range(len(labeled_anomalies)):
        print(f'  -> {labeled_anomalies["chan_id"][i]} ({i + 1} / {len(labeled_anomalies)})')
        if labeled_anomalies['spacecraft'][i] == smap_or_msl:
            # load corresponding .npy file in test and train
            # np_trn = np.load(pth + 'train/' + labeled_anomalies['chan_id'][i] + '.npy')
            # assert np_trn.shape[-1] == data_dims[smap_or_msl]
            # print(np_trn)
            # train_data.append(np_trn)

            np_tst = np.load(pth + 'test/' + labeled_anomalies['chan_id'][i] + '.npy')
            assert np_tst.shape[-1] == data_dims[smap_or_msl]
            test_data.append(np_tst)

            labs = labeled_anomalies['anomaly_sequences'][i]
            labs_s = labs.replace('[', '').replace(']', '').replace(' ', '').split(',')
            labs_i = [[int(labs_s[i]), int(labs_s[i + 1])] for i in range(0, len(labs_s), 2)]

            assert labeled_anomalies['num_values'][i] == len(np_tst)
            y_lab = np.zeros(len(np_tst))
            for sec in labs_i:
                y_lab[sec[0]:sec[1]] = 1
                total_anomaly_points += sec[1] - sec[0]
            test_label.append(y_lab)

    # print(len(test_data)) #前55个是SMAP
    # print(len(test_label))
    return test_data, test_label


if __name__ == '__main__':

    dataset='SMAP'
    data,label=create_dataset(dataset)

    data = np.concatenate(data, axis=0)  # 拼接
    label= np.concatenate(label, axis=0)
    print(data.shape)
    print(label.shape)

    # 设置时间戳长度
    N = 10000
    n = 5000

    # 设置开始和结束
    start = 0
    middle = start + N
    end = middle + n

    begin_time = time.time()

    #train
    X1=data[start:middle,:] #获取时序数据矩阵
    X2 = final_MTF(X1, N, 9) #将时序数据矩阵转换成MTF矩阵
    Y = label[start:middle]
    savemat('train_'+dataset+'.mat', {'A': X1,'B':X2,'C':Y}) #存放两个矩阵于对应文件中

    end_time = time.time()
    exe_time = round((end_time - begin_time), 2)
    print(exe_time)

    #test
    X1 = data[middle:end, :]  # 获取时序数据矩阵
    X2 = final_MTF(X1, n, 9)  # 将时序数据矩阵转换成MTF矩阵
    Y = label[middle:end]
    savemat('test_'+dataset+'.mat', {'A': X1, 'B': X2, 'C': Y})  # 存放两个矩阵于对应文件中



    # variables = io.loadmat('train_'+dataset+'.mat')
    # print(variables['A'].shape)
    # print(variables['B'].shape)
    # print(variables['C'][0].shape)
