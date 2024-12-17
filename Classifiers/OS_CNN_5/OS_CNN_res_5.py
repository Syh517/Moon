import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .OS_block import OS_block
from Classifiers.AFF import MS_CAM

def layer_parameter_list_input_change(layer_parameter_list, input_channel):
    
    new_layer_parameter_list = []
    for i, i_th_layer_parameter in enumerate(layer_parameter_list):
        if i == 0:
            first_layer_parameter = []
            for cov_parameter in i_th_layer_parameter:
                first_layer_parameter.append((input_channel,cov_parameter[1],cov_parameter[2]))
            new_layer_parameter_list.append(first_layer_parameter)
        else:
            new_layer_parameter_list.append(i_th_layer_parameter)
    return new_layer_parameter_list
        
    

class SampaddingConv1D_BN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X

    
class Res_OS_layer(nn.Module):
    def __init__(self,layer_parameter_list,out_put_channel_numebr):
        super(Res_OS_layer, self).__init__()  
        self.layer_parameter_list = layer_parameter_list
        self.net = OS_block(layer_parameter_list,False)
        self.res = SampaddingConv1D_BN(layer_parameter_list[0][0][0],out_put_channel_numebr,1)
        
    def forward(self, X):
        temp = self.net(X)
        shot_cut= self.res(X)
        block = F.relu(torch.add(shot_cut,temp))
        return block
        
        

class OS_CNN_res(nn.Module):
    def __init__(self,layer_parameter_list,n_class, n_layers, few_shot = True):
        super(OS_CNN_res, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.n_layers = n_layers
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
        new_layer_parameter_list = layer_parameter_list_input_change(layer_parameter_list, out_put_channel_numebr) #更改in_put_channel_number，其他与layer_parameter_list没什么不同


        #构建第一个net
        self.net1_1 = Res_OS_layer(layer_parameter_list,out_put_channel_numebr) #out_put_channel_numebr作为self.net的in_put_channel_number
        self.net_list_1 = []
        for i in range(self.n_layers-1):
            temp_layer = Res_OS_layer(new_layer_parameter_list,out_put_channel_numebr)
            self.net_list_1.append(temp_layer)
        self.net1 = nn.Sequential(*self.net_list_1) #把所有OS-Block顺序相连，形成网络

        # 构建第二个net
        self.net2_1 = Res_OS_layer(layer_parameter_list, out_put_channel_numebr)
        self.net_list_2 = []
        for i in range(self.n_layers - 1):
            temp_layer = Res_OS_layer(new_layer_parameter_list, out_put_channel_numebr)
            self.net_list_2.append(temp_layer)
        self.net2 = nn.Sequential(*self.net_list_2)  # 把所有OS-Block顺序相连，形成网络


        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X1, X2):
        
        temp1 = self.net1_1(X1)
        temp1 = self.net1(temp1)

        temp2 = self.net2_1(X2)
        temp2 = self.net2(temp2)

        # 通过OS-Block得到低维度特征后，进行特征的拼接
        X = torch.concat((temp1, temp2), 2) #initial method

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X
        