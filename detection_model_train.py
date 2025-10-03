from Classifiers.OS_CNN.OS_CNN_res_easy_use import OS_CNN_easy_use as OneNet_res_LGFF


import numpy as np
from scipy import io
from Classifiers.metric import metric
from Classifiers.metric import PA_K
import time
import torch
from explainer_train import pretrain




Result_log_folder = './Results_of_Models/'

dataset='SM'
model_save_path = Result_log_folder + 'trained_model_'+dataset

save_path = './ServerMachineDataset/save/'
data_train="1-4"
data_test="1-6"

if __name__ == '__main__':
    print(dataset)
    variables = io.loadmat(save_path + 'machine-'+data_train+'.mat')
    X1_train = variables['A'] #time series data X_train
    X2_train = variables['B'] #MTF matrix X_trian 
    y_train = variables['C'][0] #label y_train


    variables = io.loadmat(save_path + 'machine-'+data_test+'.mat')
    X1_test = variables['A'] #time series data X_test
    X2_test = variables['B']  # MTF matrix X_test
    y_test = variables['C'][0] #label y_test

    X1_train = np.float32(X1_train)
    X2_train = np.float32(X2_train)
    y_train = np.int64(y_train)

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



    # creat model and log save place
    model = OneNet_res_LGFF(
        device="cuda:0",  # Gpu
        model_save_path = model_save_path,
        max_epoch=5,
        paramenter_number_of_layer_list=[8 * 128 * 1, 5 * 128 * 256 + 2 * 256 * 128],
        lr=0.001
    )
    # model = torch.load(model_save_path)
    # print("model load")


    model.fit(X1_train,X2_train, y_train, X1_test, X2_test,y_test)

    torch.save(model, model_save_path)
    print("model saved")



    y_predict = model.predict(X1_test,X2_test)
    t = metric(y_test, y_predict)
    print('Precision:', t['Precision'], '\t Recall:', t['Recall'],
        '\t F1:', t['F1'], '\t Accuracy:', t['Accuracy'], '\t F1pa:', t['F1pa'],
        '\t PA_K_F1:', t['PA_K_F1'], '\t AU_PR:', t['AU_PR'])


