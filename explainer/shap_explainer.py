import shap
import numpy as np
import torch
import pandas as pd

def SelectAbnormal(X_train,y_train):
    X=[]
    for i in range(len(y_train)):
        if y_train[i]==1:
            X.append(X_train[i])
    X=np.array(X)
    return X

def Kexplainer(explainer,X_test, i):
    shap_values = explainer.shap_values(X_test[i,:].reshape(1, X_test.shape[1]))
    contribution=shap_values[1][0]
    contribution=list(contribution)
    # print(contribution)
    indexs=pd.Series(contribution).sort_values(ascending=False)#.index[:5]
    # print(indexs)
    return contribution,indexs




def Gexplainer(model,explainer,X_test, i): #输入都是图像型时间序列
    x2_test = X_test[i,:].reshape(1, X_test.shape[1])#选择需要进行异常检测的图像型时间序列
    x1_test=model.X1 #构造填充的正常数值型时间序列

    #转换成张量
    x2_test = torch.from_numpy(x2_test)
    x2_test.requires_grad = False
    x2_test = x2_test.to(model.device)

    x1_test = torch.from_numpy(x1_test)
    x1_test.requires_grad = False
    x1_test = x1_test.to(model.device)


    #升维
    if len(x1_test.shape) == 2:
        x1_test = x1_test.unsqueeze_(1)
    if len(x2_test.shape) == 2:
        x2_test = x2_test.unsqueeze_(1)


    list_test=[]
    list_test.append(x1_test)
    list_test.append(x2_test)


    shap_values= explainer.shap_values(list_test,ranked_outputs=5)
    contribution=shap_values[0][1][1]
    contribution=list(contribution[0][0])
    # print(contribution)
    indexs=pd.Series(contribution).sort_values(ascending=False)#.index[:5]
    # print(indexs)
    return contribution,indexs




def shap_explainer(model,explainer1,explainer2, X1_test, X2_test,i):
    contribution1,indexs1=Kexplainer(explainer1,X1_test,i)
    contribution2,indexs2=Gexplainer(model, explainer2,X2_test,i)
    return contribution1,contribution2,indexs1,indexs2
