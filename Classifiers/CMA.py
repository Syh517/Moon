import torch
import torch.nn as nn
import math

class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, attention_dim, device='cuda'):
        super(CrossModalAttention, self).__init__()

        self.device = device
        
        # 定义线性变换，将特征维度映射到相同的注意力维度
        self.query_transform_1 = nn.Linear(dim1, attention_dim).to(device)
        self.key_transform_2 = nn.Linear(dim2, attention_dim).to(device)
        self.value_transform_2 = nn.Linear(dim2, attention_dim).to(device)

        self.query_transform_2 = nn.Linear(dim2, attention_dim).to(device)
        self.key_transform_1 = nn.Linear(dim1, attention_dim).to(device)
        self.value_transform_1 = nn.Linear(dim1, attention_dim).to(device)

        self.scale = math.sqrt(attention_dim)

    def forward(self, feat1, feat2):
        feat1 = feat1.to(self.device)
        feat2 = feat2.to(self.device)

        # 对第一个模态生成 Q1
        Q1 = self.query_transform_1(feat1)  # [batch_size, seq_len, attention_dim]
        
        # 对第二个模态生成 K2 和 V2
        K2 = self.key_transform_2(feat2)    # [batch_size, seq_len, attention_dim]
        V2 = self.value_transform_2(feat2)  # [batch_size, seq_len, attention_dim]

        # 计算第一个模态对第二个模态的注意力权重
        attention_scores_1to2 = torch.matmul(Q1, K2.transpose(-2, -1)) / self.scale
        attention_weights_1to2 = torch.softmax(attention_scores_1to2, dim=-1)

        # 用注意力权重加权第二个模态的 V2
        F1_to_2 = torch.matmul(attention_weights_1to2, V2)  # [batch_size, seq_len, attention_dim]

        # 对第二个模态生成 Q2
        Q2 = self.query_transform_2(feat2)  # [batch_size, seq_len, attention_dim]

        # 对第一个模态生成 K1 和 V1
        K1 = self.key_transform_1(feat1)    # [batch_size, seq_len, attention_dim]
        V1 = self.value_transform_1(feat1)  # [batch_size, seq_len, attention_dim]

        # 计算第二个模态对第一个模态的注意力权重
        attention_scores_2to1 = torch.matmul(Q2, K1.transpose(-2, -1)) / self.scale
        attention_weights_2to1 = torch.softmax(attention_scores_2to1, dim=-1)

        # 用注意力权重加权第一个模态的 V1
        F2_to_1 = torch.matmul(attention_weights_2to1, V1)  # [batch_size, seq_len, attention_dim]

        return F1_to_2, F2_to_1, attention_weights_1to2, attention_weights_2to1


