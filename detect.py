import numpy as np
from scipy import io
import torch
import shap
import pandas as pd
import math
import time

from Classifiers.OS_CNN_4.OS_CNN_res_easy_use_4 import OS_CNN_easy_use as OneNet_res_MS_CAM
from Classifiers.metric import metric
from explainer.analysis import identifier
from explainer.analysis import classifier
from explainer.shap_explainer import shap_explainer
from explainer_train import get_explainer
from explainer.analysis import ground_truth
from explainer.analysis import get_interpretation



Result_log_folder = './Example_Results_of_OS_CNN_for_multivariate/'
dataset_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save2/'
dataset_name='ServerMachineDataset'
classifier_save_path = './explainer/kmeans.pkl'

dataset='SM' #MBA MSL PSM SMAP SWaT SM
model_save_path = Result_log_folder + 'trained_model_'+dataset

if __name__ == '__main__':
    variables = io.loadmat(save_path + 'machine-1-4.mat')
    # variables = io.loadmat('./Data/'+dataset+'Dataset/train_'+dataset+ '.mat')
    X1_train = variables['A'] #读取时序数据X_train
    X2_train = variables['B'] #读取MTF矩阵X_trian
    y_train = variables['C'][0] #读取标签y_train

    X1_train = np.float32(X1_train)
    X2_train = np.float32(X2_train)
    y_train = np.int64(y_train)

    variables = io.loadmat(save_path + 'machine-1-6.mat')
    # variables = io.loadmat('./Data/'+dataset+'Dataset/test_'+dataset+ '.mat')
    X1_test = variables['A'] #读取时序数据X_test
    X2_test = variables['B']  # 读取MTF矩阵X_test
    y_test = variables['C'][0] #读取标签y_test

    X1_test=np.float32(X1_test)
    X2_test = np.float32(X2_test)
    y_test = np.int64(y_test)



    print('train data 1 shape', X1_train.shape)
    print('train data 2 shape', X2_train.shape)
    print('train label shape', y_train.shape)
    print('test data 1 shape', X1_test.shape)
    print('test data 2 shape', X2_test.shape)
    print('test label shape', y_test.shape)
    print('unique train label', np.unique(y_train))
    print('unique test label', np.unique(y_test))

    model = OneNet_res_MS_CAM(
        device="cuda:0",  # Gpu
        model_save_path = model_save_path,
        max_epoch=100,
        paramenter_number_of_layer_list=[8 * 128 * 1, 5 * 128 * 256 + 2 * 256 * 128],  # 两种X_train参数是一样的
        lr=0.005
    )
    model = torch.load(model_save_path)

    y_predict = model.predict(X1_test, X2_test)
    y_predict = np.int64(y_predict)
    # print('correct:', y_test)
    # print('predict:', y_predict)
    print(np.unique( y_predict))




    t = metric(y_test, y_predict)
    print('Precision:', t['Precision'], '\t Recall:', t['Recall'],
          '\t F1:', t['F1'], '\t Accuracy:', t['Accuracy'], '\t F1pa:', t['F1pa'],
          '\t PA_K_F1:', t['PA_K_F1'], '\t AU_PR:', t['AU_PR'])

    #获得正常数据normal
    normal_index=-1
    for i in range(len(y_test)):
        if y_test[i] == 0:
            normal_index = i
            break
    if normal_index != -1:
        normal = X1_test[normal_index]
    else:
        print("Error:Cannot get normal time series data!")


    exp1, exp2 = get_explainer(model, X1_train, X2_train, y_train)
    w = 0.7
    print("!!!")

    # for i in range(y_predict.shape[0]):
    #     if y_predict[i]==1:
    #         contribution1, contribution2, indexs1, indexs2 = shap_explainer(model, exp1, exp2, X1_test, X2_test, i)
    #         s = sum(contribution1)
    #         A_C = []
    #         for j in range(len(contribution1)):
    #             A_C.append(w * contribution1[j] + ((1 - w) * s * contribution2[j]) / len(contribution1))
    #
    #         indexs3 = pd.Series(A_C).sort_values(ascending=False)
    #         order = list(indexs3.index)
    #
    #         X = X1_test[i, :].reshape(1, X1_test.shape[1])
    #         As = identifier(model, X, order,normal)
    #         label = classifier(X, A_C, dataset)
    #
    #         print('异常变量为:',As)
    #         print('异常归类为:',label)


    #获得SMD解释异常的变量及初始化相关参数
    start, end, variable_list = get_interpretation()
    hit_P = 0
    count = 0


    if dataset=='SM' :
        for i in range(y_predict.shape[0]):
            if y_predict[i]==1 and y_test[i]==1:
                count += 1
                print(i)
                print('count:',count)
                
                begin_time = time.time()
                
                contribution1, contribution2, indexs1, indexs2 = shap_explainer(model, exp1, exp2, X1_test, X2_test, i)
                s = sum(contribution1)
                A_C = []
                for j in range(len(contribution1)):
                    A_C.append(w * contribution1[j] + ((1 - w) * s * contribution2[j]) / len(contribution1))
                    
                end_time_1 = time.time()
                print(round(end_time_1 - begin_time, 3))

                indexs3 = pd.Series(A_C).sort_values(ascending=False)
                order = list(indexs3.index)

                As=order
                print('As:',As)

                Gt=ground_truth(start,end,variable_list,i)
                print('Gt:',Gt)

                end_time_2 = time.time()
                print(round(end_time_2 - begin_time, 3))

                hit = 0
                P=1.5 #P=1 or P=1.5
                n=math.ceil(P*len(Gt))
                n=min(n,len(As))
                for j in range(n):
                    if As[j] in Gt:
                        hit += 1
                hit=round(hit/len(Gt),4)
                hit_P+=hit
                print('hit_P:',round(hit_P/count,4))
                print('\n')

        if count:
            hit_P=round(hit_P/count,4)
        print('hit_P:',hit_P)






