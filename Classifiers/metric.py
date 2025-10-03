import os
import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy




def adjust_predicts(label, pred):
    if len(label) != len(pred):
        raise ValueError("label and pred must have the same length")
    
    actual = label #
    adjust_pred = copy.deepcopy(pred)  
    
    anomaly_state = False
    anomaly_count = 0
    
    for i in range(len(label)):
        if actual[i] and adjust_pred[i] and not anomaly_state: 
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]: 
                    break
                else:
                    if not adjust_pred[j]: 
                        adjust_pred[j] = 1
        elif not actual[i]: 
            anomaly_state = False
        if anomaly_state:
            adjust_pred[i] = 1
    
    return adjust_pred
    

def get_P_R_A(y_true,y_pred):
    TP=0
    TN=0
    FP=0
    FN=0

    for i in range(len(y_true)):
        if y_true[i] ==0 and y_pred[i] ==0:
            TP+=1
        elif y_true[i] ==1 and y_pred[i] ==1:
            TN+=1
        elif y_true[i] ==1 and y_pred[i] ==0:
            FP+=1
        else: # y_true[i] ==0 and y_pred[i] ==1
            FN+=1
    


    if TP+FP==0:
        Precision=0
    else:
        Precision=TP/(TP+FP)

    if TP +FN ==0:
        Recall=0
    else:
        Recall=TP/(TP+FN)

    Accuracy=(TP+TN)/(TP+TN+FP+FN)

    
    return Precision, Recall, Accuracy

def PA_K(y_pred):
    K=0.8
    n_segment=20
    n_group=int(np.ceil(len(y_pred)/n_segment))

    y_pred_PA_K=y_pred.copy()

    for i in range(n_group):
        start=i*n_segment
        end=min((i+1)*n_segment,len(y_pred))

        count=0
        for j in range(start,end):
            if y_pred_PA_K[j]==1:
                count+=1

        
        if count>=K*n_segment:
            for j in range(start,end):
                y_pred_PA_K[j]=1
        else:
            for j in range(start,end):
                y_pred_PA_K[j]=0

    return y_pred_PA_K

    

def metric(y_true,y_pred):

    Precision, Recall, Accuracy=get_P_R_A(y_true,y_pred)

    if Precision==0 or Recall==0 or Precision+Recall==0:
        F1=0
    else:
        F1=2*Precision*Recall/(Precision+Recall)

    
    #Adjust abnormal labels
    adjust_pred = adjust_predicts(y_true,y_pred)
    p, r, a=get_P_R_A(y_true,adjust_pred)


    if p==0 or r==0 or p+r==0:
        F1pa=0
    else:
        F1pa=2*p*r/(p+r)

    #PA%K
    y_pred_PA_K=PA_K(y_pred)
    p, r, a=get_P_R_A(y_true,y_pred_PA_K)

    if p==0 or r==0 or p+r==0:
        PA_K_F1=0
    else:
        PA_K_F1=2*p*r/(p+r)


    AU_PR=metrics.average_precision_score(y_true, y_pred)


    t= {}
    t['Precision']=Precision
    t['Recall']=Recall
    t['F1']=F1
    t['Accuracy'] =Accuracy
    t['F1pa']=F1pa
    t['PA_K_F1']=PA_K_F1
    t['AU_PR']=AU_PR
    t['adjust_pred']=adjust_pred


    return t

