import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from Classifiers.AFF import MS_CAM
from Classifiers.LGFF import LGFF
from Classifiers.CMA import CrossModalAttention

class LSTM(nn.Module):
    def __init__(self, input_dim, n_class, hidden_dim=128, n_layers=2, device='cuda:0'):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        self.attention_dim = 64
        self.cma = None

        self.fusion_mode=None

        self.averagepool=None

        self.hidden = nn.Linear(hidden_dim, n_class)

    def forward(self, X1, X2):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, X1.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, X1.size(0), self.hidden_dim).to(self.device)
        
        temp1, _ = self.lstm(X1, (h0, c0)) 
        temp2, _ = self.lstm(X2, (h0, c0)) 


        # print(temp1.shape)

        # 跨模态注意力机制
        dim_num, dim_img = temp1.shape[-1], temp2.shape[-1]
        if self.cma is None:
            self.cma = CrossModalAttention(dim_num, dim_img, self.attention_dim)
        F_num_to_img, F_img_to_num, attn_weights_num_to_img, attn_weights_img_to_num =self.cma(temp1,temp2)
        temp1=F_num_to_img
        temp2=F_img_to_num

        # print(temp1.shape)


        #特征融合
        if self.fusion_mode is None:
            output_channel_1=temp1.shape[1]
            output_channel_2=temp2.shape[1]
            self.fusion_mode=LGFF(output_channel_1, output_channel_2, 1, bias=False).to(self.device)
  
        X = torch.concat((temp1, temp2), 2)
        X = X.unsqueeze_(2)
        X = self.fusion_mode(X)
        X = X.squeeze_(2)

        out_put_channel_numebr = X.shape[1]
        if self.averagepool is None:
            self.averagepool = nn.AvgPool1d(out_put_channel_numebr).to(self.device)
        

        X = self.averagepool(X)
        X = X.squeeze_(-1)


        # Decode the hidden state of the last time step
        X = self.hidden(X[:, -1, :])
        return X
