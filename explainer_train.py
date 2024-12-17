from sklearn.cluster import KMeans
import numpy as np
from scipy import io
import joblib
import torch
import shap
import time

from explainer.shap_explainer import shap_explainer


def SelectAbnormal(X1,X2,y):
    X1_=[]
    X2_=[]
    for i in range(len(y)):
        if y[i]==1:
            X1_.append(X1[i])
            X2_.append(X2[i])
    X1_=np.array(X1_)
    X2_ = np.array(X2_)
    return X1_,X2_

def get_explainer(model,X1_train,X2_train,y_train):
    #获取异常数据
    X1_train,X2_train = SelectAbnormal(X1_train,X2_train,y_train)

    ##### 训练identifier
    ### kernel
    KX1_train = X1_train
    explainer1 = shap.KernelExplainer(model.predict_ts, KX1_train, link="logit")

    ### Gradient
    GX2_train = X2_train
    n = GX2_train.shape[0]
    GX1_train = np.tile(model.X1, (n, 1))  # 构造填充的正常数值型时间序列矩阵

    # 转换成张量
    GX1_train = torch.from_numpy(GX1_train)
    GX1_train.requires_grad = False
    GX1_train = GX1_train.to(model.device)
    GX2_train = torch.from_numpy(GX2_train)
    GX2_train.requires_grad = False
    GX2_train = GX2_train.to(model.device)

    # 升维
    if len(GX1_train.shape) == 2:
        GX1_train = GX1_train.unsqueeze_(1)
    if len(GX2_train.shape) == 2:
        GX2_train = GX2_train.unsqueeze_(1)

    list_train = []
    list_train.append(GX1_train)
    list_train.append(GX2_train)

    explainer2 = shap.GradientExplainer(model.OS_CNN, data=list_train, local_smoothing=0)

    return explainer1, explainer2

def pretrain(model, dataset,X1_train, X2_train, y_train, n = None):

    #explainer和获取异常数据
    exp1,exp2 = get_explainer(model, X1_train, X2_train,y_train)
    X1_train,X2_train = SelectAbnormal(X1_train,X2_train,y_train)


    ##### 训练classifier
    # 处理X1_train
    print("The number of abnormal data is:",X1_train.shape[0])
    if n == None:
        n=X1_train.shape[0]
    else:
        n=min(X1_train.shape[0],n)

    print("The number of train data is:", n)
    CX1_train=X1_train[:n,:]
    CX2_train = X2_train[:n, :]
    w = 0.8
    for i in range(CX1_train.shape[0]):
        contribution1, contribution2, indexs1, indexs2 = shap_explainer(model, exp1, exp2, CX1_train, CX2_train, i)
        s = sum(contribution1)
        A_C = []

        for j in range(len(contribution1)):
            A_C.append(w * contribution1[j] + ((1 - w) * s * contribution2[j]) / len(contribution1))

        for k in range(len(A_C)):
            CX1_train[i, k] = CX1_train[i, k] * A_C[k]

    # 训练kmeans-model
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=0,n_init='auto')
    kmeans.fit(CX1_train)
    joblib.dump(kmeans, './explainer/'+dataset+'_kmeans.pkl')


    # labels = kmeans.predict(X_test)
    # centroids = kmeans.cluster_centers_


