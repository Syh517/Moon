import shap
import numpy as np
import torch
import pandas as pd
import math
from matplotlib import pyplot as plt
shap.initjs()

def SelectAbnormal(X_train,y_train):
    X=[]
    for i in range(len(y_train)):
        if y_train[i]==1:
            X.append(X_train[i])
    X=np.array(X)
    return X

def Kexplainer(model,X_train, y_train,X_test, i):
    X=SelectAbnormal(X_train, y_train)

    explainer = shap.KernelExplainer(model.predict_ts, X, link="logit")
    shap_values = explainer.shap_values(X_test[i,:].reshape(1, X_test.shape[1]))
    contribution=shap_values[1][0]
    contribution=list(contribution)
    # print(contribution)
    indexs=pd.Series(contribution).sort_values(ascending=False)#.index[:5]
    # print(indexs)
    return contribution,indexs

    # shap.summary_plot(shap_values[1], X_test[i].reshape(1, X_test[0].shape[0]), max_display=38,
    #                   feature_names=feature_names, plot_type="bar")



def Kexplainer2(model,X_train, y_train,X_test, i):
    X=SelectAbnormal(X_train, y_train)
    feature_names = ['Feature 0', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5',
                     'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11',
                     'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15', 'Feature 16',
                     'Feature 17', 'Feature 18', 'Feature 19', 'Feature 20', 'Feature 21',
                     'Feature 22', 'Feature 23', 'Feature 24', 'Feature 25', 'Feature 26',
                     'Feature 27', 'Feature 28', 'Feature 29', 'Feature 30', 'Feature 31',
                     'Feature 32', 'Feature 33', 'Feature 34', 'Feature 35', 'Feature 36', 'Feature 37']

    explainer = shap.KernelExplainer(model.predict_img, X, link="logit",feature_names=feature_names)
    shap_values = explainer.shap_values(X_test[i,:].reshape(1, X_test.shape[1]))

    shap.summary_plot(shap_values, X_test[i].reshape(1, X_test[0].shape[0]), max_display=38,
                      feature_names=feature_names, plot_type="bar")

def Gexplainer(model,X_train, y_train,X_test, i): #输入都是图像型时间序列
    X2_train=SelectAbnormal(X_train, y_train) #选择异常的MTF图像型时间序列矩阵
    n = X2_train.shape[0]
    y_train= np.tile([1], n) #构造对应的异常标签
    X1_train=np.tile(model.X1, (n,1)) #构造填充的正常数值型时间序列矩阵

    x2_test = X_test[i,:].reshape(1, X_test.shape[1])#选择需要进行异常检测的图像型时间序列
    x1_test=model.X1 #构造填充的正常数值型时间序列

    #转换成张量
    X2_train = torch.from_numpy(X2_train)
    X2_train.requires_grad = False
    X2_train = X2_train.to(model.device)

    X1_train = torch.from_numpy(X1_train)
    X1_train.requires_grad = False
    X1_train = X1_train.to(model.device)

    x2_test = torch.from_numpy(x2_test)
    x2_test.requires_grad = False
    x2_test = x2_test.to(model.device)

    x1_test = torch.from_numpy(x1_test)
    x1_test.requires_grad = False
    x1_test = x1_test.to(model.device)


    #升维
    if len(X1_train.shape) == 2:
        X1_train = X1_train.unsqueeze_(1)
        x1_test = x1_test.unsqueeze_(1)

    if len(X2_train.shape) == 2:
        X2_train = X2_train.unsqueeze_(1)
        x2_test = x2_test.unsqueeze_(1)


    list_train=[]
    list_train.append(X1_train)
    list_train.append(X2_train)

    list_test=[]
    list_test.append(x1_test)
    list_test.append(x2_test)
    explainer = shap.GradientExplainer(model.OS_CNN, data=list_train, local_smoothing=0)
    shap_values= explainer.shap_values(list_test,ranked_outputs=5)
    contribution=shap_values[0][1][1]
    contribution=list(contribution[0][0])
    # print(contribution)
    indexs=pd.Series(contribution).sort_values(ascending=False)#.index[:5]
    # print(indexs)
    return contribution,indexs






def shap_exp(model,X1_train,X2_train, y_train, X1_test, X2_test, i):
    contribution1,indexs1=Kexplainer(model,X1_train,y_train,X1_test,i)
    # Kexplainer2(model, X2_train, y_train, X2_test, i)
    contribution2,indexs2=Gexplainer(model, X2_train, y_train, X2_test, i)
    return contribution1,contribution2,indexs1,indexs2
