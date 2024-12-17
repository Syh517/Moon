import shap_explainer
import math
from matplotlib import pyplot as plt
from IPython.display import (display, display_html, display_png, display_svg)
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset



def SelectAbnormal(X_train,y_train):
    X=[]
    for i in range(len(y_train)):
        if y_train[i]==1:
            X.append(X_train[i])
    X=np.array(X)
    return X

def Gexplainer(model,X_train, y_train,X_test, i):
    X_train=SelectAbnormal(X_train, y_train)
    n = X_train.shape[0]
    y_train= np.tile([1], n)
    x_test=X_test[i:i + 1, ]

    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(model.device)

    x_test = torch.from_numpy(x_test)
    x_test.requires_grad = False
    x_test = x_test.to(model.device)

    y_train = torch.from_numpy(y_train).to(model.device)


    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
        x_test=x_test.unsqueeze_(1)

    feature_names = ['Feature 0', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5',
                     'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11',
                     'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15', 'Feature 16',
                     'Feature 17', 'Feature 18', 'Feature 19', 'Feature 20', 'Feature 21',
                     'Feature 22', 'Feature 23', 'Feature 24', 'Feature 25', 'Feature 26',
                     'Feature 27', 'Feature 28', 'Feature 29', 'Feature 30', 'Feature 31',
                     'Feature 32', 'Feature 33', 'Feature 34', 'Feature 35', 'Feature 36', 'Feature 37']
    explainer = shap.GradientExplainer(model.model, X_train, local_smoothing=0)
    shap_values = explainer.shap_values(x_test)
    print(shap_values)

    x_test = x_test.squeeze(1)
    for i in range(len(shap_values)):
        shap_values[i]=shap_values[i].squeeze(1)


    shap.summary_plot(shap_values, x_test,max_display=38,feature_names=feature_names, plot_type="bar")



