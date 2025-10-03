import numpy as np
from scipy import io
import torch
import shap
import pandas as pd
import math
import time

from Classifiers.OS_CNN.OS_CNN_res_easy_use import OS_CNN_easy_use as OneNet_res_LGFF
from Classifiers.metric import metric
from explainer.shap_explainer import shap_explainer
from explainer_train import get_explainer
from explainer.analysis import ground_truth
from explainer.analysis import get_interpretation
import numpy as np
from sklearn.metrics import ndcg_score
def ndcg(A_C, Gt, p):
    ndcg_scores = []
    a = A_C
    n= len(a)
    l = [1 if (i+1) in Gt else 0 for i in range(n)]

    a = np.array(a)
    l = np.array(l)
    
    if Gt:
        k_p = round(p * len(Gt))
        hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
        ndcg_scores.append(hit)
    # res = np.mean(ndcg_scores)
    return hit



Result_log_folder = './Results_of_Models/'
dataset_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save/'

dataset='SM' 
model_save_path = Result_log_folder + 'trained_model_'+dataset

data_train="1-4"
data_test="1-6"


if __name__ == '__main__':
    begin = time.time()

    print(dataset)
    variables = io.loadmat(save_path + 'machine-'+data_train+'.mat')
    X1_train = variables['A'] #time series data X_train
    X2_train = variables['B'] #MTF matrix X_trian
    y_train = variables['C'][0] #label y_train

    X1_train = np.float32(X1_train)
    X2_train = np.float32(X2_train)
    y_train = np.int64(y_train)

    variables = io.loadmat(save_path + 'machine-'+data_test+'.mat')
    X1_test = variables['A'] #time series data X_test
    X2_test = variables['B']  # MTF matrix X_test
    y_test = variables['C'][0] #label y_test

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


    torch.cuda.empty_cache()
    model = OneNet_res_LGFF(
        device="cuda:0",  # Gpu
        model_save_path = model_save_path,
        max_epoch=5,
        paramenter_number_of_layer_list=[8 * 128 * 1, 5 * 128 * 256 + 2 * 256 * 128], 
        lr=0.001
    )
    model = torch.load(model_save_path)

    # model.fit(X1_train,X2_train, y_train, X1_test, X2_test,y_test)
    # torch.save(model, model_save_path2)
    # print("model saved")



    y_predict = model.predict(X1_test, X2_test)
    y_predict = np.int64(y_predict)
    print(np.unique( y_predict))

    t = metric(y_test, y_predict)
    print('Precision:', t['Precision'], '\t Recall:', t['Recall'],
        '\t F1:', t['F1'], '\t Accuracy:', t['Accuracy'], '\t F1pa:', t['F1pa'],
        '\t PA_K_F1:', t['PA_K_F1'], '\t AU_PR:', t['AU_PR'])



    # get normal data normal
    normal_index=-1
    for i in range(len(y_test)):
        if y_test[i] == 0:
            normal_index = i
            break
    if normal_index != -1:
        normal = X1_test[normal_index]
    else:
        print("Error:Cannot get normal time series data!")


    exp1, exp2 = get_explainer(model, X1_train, X2_train, y_train, r=0.1)  ####################
    w = 0
    adjust_pred=t['adjust_pred']



    start, end, variable_list = get_interpretation('./ServerMachineDataset/interpretation_label/machine-'+data_test+'.txt')
    hit_1 = 0
    hit_2 = 0
    count = 0

    ndcg_100 = []
    ndcg_150 = []


    if dataset=='SM' :
        for i in range(y_predict.shape[0]):
            # if adjust_pred[i]==1 and y_test[i]==1:
            if y_predict[i]==1 and y_test[i]==1:
                count += 1
                print(i)
                print('count:',count)
                
                
                contribution1, contribution2, indexs1, indexs2 = shap_explainer(model, exp1, exp2, X1_test, X2_test, i)
                s = sum(contribution1)
                A_C = []
                for j in range(len(contribution1)):
                    A_C.append(w * contribution1[j] + ((1 - w) * s * contribution2[j]) / len(contribution1))
                
                    

                indexs3 = pd.Series(A_C).sort_values(ascending=False)
                order = list(indexs3.index)
                for k in range(len(order)):
                    order[k]=order[k]+1

                indexs1 = pd.Series(contribution1).sort_values(ascending=False)
                order1 = list(indexs1.index)
                for k in range(len(order1)):
                    order1[k]=order1[k]+1
                # print('kernel:',order1)

                indexs2 = pd.Series(contribution2).sort_values(ascending=False)
                order2 = list(indexs2.index)
                for k in range(len(order2)):
                    order2[k]=order2[k]+1
                # print('gradient:',order2)

                As=order
                print('As:',As)

                Gt=ground_truth(start,end,variable_list,i) 
                print('Gt:',Gt)

                ndcg_1=ndcg(A_C, Gt, 1)
                ndcg_100.append(ndcg_1)
                print("NDCG@100%", ndcg_1, np.mean(ndcg_100))
                print('\n')


                ndcg_2=ndcg(A_C, Gt, 1.5)
                ndcg_150.append(ndcg_2)
                print("NDCG@150%", ndcg_2, np.mean(ndcg_150))
                print('\n')



                hit1 = 0
                P=1 
                n1=math.ceil(P*len(Gt))
                n1=min(n1,len(As))
                for j in range(n1):
                    if As[j] in Gt:
                        hit1 += 1
                hit1=round(hit1/len(Gt),4)
                hit_1+=hit1
                print('hit_1:',hit1)
                print('hit_1_total:',hit_1)
                print('hit_1:',round(hit_1/count,4))
                print('\n')

                hit2 = 0
                P=1.5 
                n2=math.ceil(P*len(Gt))
                n2=min(n2,len(As))
                for j in range(n2):
                    if As[j] in Gt:
                        hit2 += 1
                hit2=round(hit2/len(Gt),4)
                hit_2+=hit2
                print('hit_2:',hit2)
                print('hit_2_total:',hit_2)
                print('hit_2:',round(hit_2/count,4))
                print('\n')













