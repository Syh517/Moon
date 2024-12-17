import torch
import torch.nn as nn
import torch.nn.functional as F
from Classifiers.LGFF import LGFF

class FCN(nn.Module):
    def __init__(self,input_shape,n_class,RF_size):
        super(FCN, self).__init__()
        hidden_layer_1 = 128
        kernel_size = int(RF_size/2)
        self.padding1 = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=hidden_layer_1, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layer_1)
        self.relu1 = nn.ReLU()

        kernel_size = int(RF_size*5/16)
        hidden_layer_2 = hidden_layer_1*2
        self.padding2 = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_layer_1, out_channels=hidden_layer_2, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layer_2)
        self.relu2 = nn.ReLU()

        hidden_layer_3 = hidden_layer_1
        kernel_size = int(RF_size*3/16)
        self.padding3 = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv3 = torch.nn.Conv1d(in_channels=hidden_layer_2, out_channels=hidden_layer_3, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_layer_3)
        self.relu3 = nn.ReLU()

        self.averagepool = nn.AvgPool1d(kernel_size = input_shape)

        self.hidden = nn.Linear(hidden_layer_3, n_class)


    def forward(self, X1,X2):
        X1 = self.padding1(X1)
        X1 = self.conv1(X1)
        X1 = self.bn1(X1)
        X1 = self.relu1(X1)

        X1 = self.padding2(X1)
        X1 = self.conv2(X1)
        X1 = self.bn2(X1)
        X1 = self.relu2(X1)

        X1 = self.padding3(X1)
        X1 = self.conv3(X1)
        X1 = self.bn3(X1)
        X1 = self.relu3(X1)

        X2 = self.padding1(X2)
        X2 = self.conv1(X2)
        X2 = self.bn1(X2)
        X2 = self.relu1(X2)

        X2 = self.padding2(X2)
        X2 = self.conv2(X2)
        X2 = self.bn2(X2)
        X2 = self.relu2(X2)

        X2 = self.padding3(X2)
        X2 = self.conv3(X2)
        X2 = self.bn3(X2)
        X2 = self.relu3(X2)

        # 新的MS_CAM特征融合方法
        X = torch.concat((X1, X2), 2)

        X = self.averagepool(X)
        X = X.squeeze_(-1)
        
        X = self.hidden(X)
        return X
        