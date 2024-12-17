import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from Classifiers.AFF import MS_CAM

from Classifiers.LGFF import LGFF

# class SampaddingConv1D_BN(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size):
#         super(SampaddingConv1D_BN, self).__init__()
#         self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
#         self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         self.bn = nn.BatchNorm1d(num_features=out_channels)
        
#     def forward(self, X):
#         X = self.padding(X)
#         X = self.conv1d(X)
#         X = self.bn(X)
#         return X
    
# class build_layer_with_layer_parameter(nn.Module):
#     def __init__(self,layer_parameters): 
#         super(build_layer_with_layer_parameter, self).__init__()
#         self.conv_list = nn.ModuleList()
        
#         for i in layer_parameters:
#             conv = SampaddingConv1D_BN(i[0],i[1],i[2])
#             self.conv_list.append(conv)
    
#     def forward(self, X):
        
#         conv_result_list = []
#         for conv in self.conv_list:
#             conv_result = conv(X)
#             conv_result_list.append(conv_result)
            
#         result = F.relu(torch.cat(tuple(conv_result_list), 1))
#         return result



def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels) #批规范化
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    
    
class OS_CNN(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True): #设置架构
        super(OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list_1 = []
        self.layer_list_2 = []

        #构建第一个net
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list_1.append(layer)
        self.net_1 = nn.Sequential(*self.layer_list_1) #把所有层顺序相连，形成一个OS-Block

        # 构建第二个net
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list_2.append(layer)
        self.net_2 = nn.Sequential(*self.layer_list_2)  # 把所有层顺序相连，形成一个OS-Block
            
        self.averagepool = nn.AdaptiveAvgPool1d(1) #1d的自适应平均池化层，可以将最后一维变为标量

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1]
            # print(final_layer_parameters) #(225,280,1) (225,280,2)


        #MS_CAM特征融合
        self.fusion_mode = MS_CAM(out_put_channel_numebr)

        self.hidden = nn.Linear(out_put_channel_numebr, n_class) #全连接层，out_put_channel_number在concatenate后有变化

    def forward(self, X1, X2): #train时调用
        
        X1 = self.net_1(X1) #OS-BLock获得低维度特征
        X2 = self.net_2(X2)  # OS-BLock获得低维度特征

        #通过OS-Block得到低维度特征后，进行特征的拼接
        # X=torch.concat((X1,X2),2) #initial method

        # 新的MS_CAM特征融合方法
        X = torch.concat((X1, X2), 2)
        X = X.unsqueeze_(2)
        X = self.fusion_mode(X)
        X=X.squeeze_(2)

        X = self.averagepool(X) #Global average pooling
        X = X.squeeze_(-1) #若X最后一维是1维，就把X压缩

        if not self.few_shot: #few_shot==false,这一步肯定执行
            X = self.hidden(X) #执行全连接层，得到分类结果
        return X
        