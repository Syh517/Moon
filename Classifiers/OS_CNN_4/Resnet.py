import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from Classifiers.AFF import MS_CAM
from Classifiers.LGFF import LGFF
from Classifiers.CMA import CrossModalAttention

class Resnet(nn.Module):
    def __init__(self,n_class, device='cuda:0'):
        super(Resnet, self).__init__()
        n_feature_maps = 64
        self.n_class = n_class
        self.device = device

        # BLOCK 1
        kernel_size = 8
        self.padding1_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_x = torch.nn.Conv1d(in_channels=1, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_x = nn.BatchNorm1d(num_features=n_feature_maps)
        self.relu1_x = nn.ReLU()

        kernel_size = 5
        self.padding1_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_y = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_y = nn.BatchNorm1d(num_features=n_feature_maps)
        self.relu1_y = nn.ReLU()

        kernel_size = 3
        self.padding1_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv1_z = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=kernel_size)
        self.bn1_z = nn.BatchNorm1d(num_features=n_feature_maps)


        self.conv1_sy = torch.nn.Conv1d(in_channels=1, out_channels=n_feature_maps, kernel_size=1)
        self.bn1_sy = nn.BatchNorm1d(num_features=n_feature_maps)
        

        self.block1 = nn.Sequential(self.padding1_x,self.conv1_x,self.bn1_x,self.relu1_x,\
                                    self.padding1_y,self.conv1_y,self.bn1_y,self.relu1_y,\
                                    self.padding1_z,self.conv1_z,self.bn1_z,)



        # BLOCK 2
        kernel_size = 8
        self.padding2_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_x = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_x = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu2_x = nn.ReLU()

        kernel_size = 5
        self.padding2_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_y = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_y = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu2_y = nn.ReLU()

        kernel_size = 3
        self.padding2_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv2_z = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn2_z = nn.BatchNorm1d(num_features=n_feature_maps*2)

        self.conv2_sy = torch.nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=1)
        self.bn2_sy = nn.BatchNorm1d(num_features=n_feature_maps*2)

        self.block2 = nn.Sequential(self.padding2_x,self.conv2_x,self.bn2_x,self.relu2_x,\
                                    self.padding2_y,self.conv2_y,self.bn2_y,self.relu2_y,\
                                    self.padding2_z,self.conv2_z,self.bn2_z)

        # BLOCK 3
        kernel_size = 8
        self.padding3_x = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_x = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_x = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu3_x = nn.ReLU()

        kernel_size = 5
        self.padding3_y = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_y = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_y = nn.BatchNorm1d(num_features=n_feature_maps*2)
        self.relu3_y = nn.ReLU()

        kernel_size = 3
        self.padding3_z = nn.ConstantPad1d((int((kernel_size-1)/2), math.ceil((kernel_size-1)/2)), 0)
        self.conv3_z = torch.nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=kernel_size)
        self.bn3_z = nn.BatchNorm1d(num_features=n_feature_maps*2)

        self.bn3_sy = nn.BatchNorm1d(num_features=n_feature_maps * 2)


        self.block3 = nn.Sequential(self.padding3_x,self.conv3_x,self.bn3_x,self.relu3_x,\
                                    self.padding3_y,self.conv3_y,self.bn3_y,self.relu3_y,\
                                    self.padding3_z,self.conv3_z,self.bn3_z,\
                                    self.bn3_sy)



        self.attention_dim = 64
        self.cma = None

        self.fusion_mode=None

        
        # self.fusion_mode = MS_CAM(out_put_channel_numebr)

        # ffn_expansion_factor = 1
        # self.fusion_mode2 = LGFF(out_put_channel_numebr, out_put_channel_numebr, ffn_expansion_factor, bias=False)


        # self.averagepool = nn.AvgPool1d(128)
        # self.hidden = nn.Linear(n_feature_maps*2, n_class)
        self.averagepool=None
        self.n_class=n_class
        self.hidden=None


    def forward(self, X1, X2):
        #block1
        temp1_1 = self.block1(X1)
        shot_cut_X1 = self.conv1_sy(X1)
        shot_cut_X1 = self.bn1_sy(shot_cut_X1)
        block1_1 = torch.add(shot_cut_X1,temp1_1)
        block1_1 = F.relu(block1_1)

        # block2
        temp2_1 = self.block2(block1_1)
        shot_cut_block1_1 = self.conv2_sy(block1_1)
        shot_cut_block1_1 = self.bn2_sy(shot_cut_block1_1)
        block2_1 = torch.add(shot_cut_block1_1,temp2_1)
        block2_1 = F.relu(block2_1)

        # block3
        temp3_1 = self.block3(block2_1)
        shot_cut_block2_1 = self.bn3_sy(block2_1)
        block3_1 = torch.add(shot_cut_block2_1, temp3_1)
        block3_1 = F.relu(block3_1)

        



        #block1
        temp1_2 = self.block1(X2)
        shot_cut_X2 = self.conv1_sy(X2)
        shot_cut_X2 = self.bn1_sy(shot_cut_X2)
        block1_2 = torch.add(shot_cut_X2,temp1_2)
        block1_2 = F.relu(block1_2)

        # block2
        temp2_2 = self.block2(block1_2)
        shot_cut_block1_2 = self.conv2_sy(block1_2)
        shot_cut_block1_2 = self.bn2_sy(shot_cut_block1_2)
        block2_2 = torch.add(shot_cut_block1_2,temp2_2)
        block2_2 = F.relu(block2_2)

        # block3
        temp3_2 = self.block3(block2_2)
        shot_cut_block2_2 = self.bn3_sy(block2_2)
        block3_2 = torch.add(shot_cut_block2_2, temp3_2)
        block3_2 = F.relu(block3_2)



        temp1 = block3_1
        temp2 = block3_2


        # 跨模态注意力机制
        dim_num, dim_img = temp1.shape[-1], temp2.shape[-1]
        if self.cma is None:
            self.cma = CrossModalAttention(dim_num, dim_img, self.attention_dim).to(self.device)
        F_num_to_img, F_img_to_num, attn_weights_num_to_img, attn_weights_img_to_num =self.cma(temp1,temp2)
        temp1=F_num_to_img
        temp2=F_img_to_num


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
            self.hidden = nn.Linear(out_put_channel_numebr, self.n_class).to(self.device)
        
        X = self.averagepool(X)
        X = X.squeeze_(-1)

        X = self.hidden(X)

        return X