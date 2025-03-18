import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv, BatchNorm


# 2. StableGAT - 3 layers
class StableGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads1=8, heads2=4, dropout=0.5, edge_dropout=0.2):
        super(StableGAT, self).__init__()
        # 第一层
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads1, concat=True)
        self.norm1 = LayerNorm(hidden_dim * heads1)

        # 第二层
        self.conv2 = GATConv(hidden_dim * heads1, hidden_dim, heads=heads2, concat=True)
        self.norm2 = LayerNorm(hidden_dim * heads2)

        # 输出层
        self.conv3 = GATConv(hidden_dim * heads2, output_dim, heads=1, concat=False)
        self.norm3 = LayerNorm(output_dim)

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        # 进行边 dropout
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        # 第一层: 卷积 -> 归一化 -> 激活 -> dropout
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层: 卷积 -> 归一化 -> 激活 -> dropout
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层: 卷积 -> 归一化 -> softmax
        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        return F.log_softmax(x, dim=1)

# 3. BalancedGAT - 4 layers
class BalancedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads1=8, heads2=4, dropout=0.5, edge_dropout=0.2):
        super(BalancedGAT, self).__init__()
        # 增加模型容量和表达能力
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads1, concat=True)
        self.norm1 = BatchNorm(hidden_dim * heads1)

        # 增加一个中间层来提高特征提取能力
        self.conv_mid = GATConv(hidden_dim * heads1, hidden_dim * heads1, heads=1, concat=False)
        self.norm_mid = BatchNorm(hidden_dim * heads1)

        self.conv2 = GATConv(hidden_dim * heads1, hidden_dim, heads=heads2, concat=True)
        self.norm2 = BatchNorm(hidden_dim * heads2)

        # 为每个类别专门设置注意力机制
        self.class_attention = nn.ModuleList([
            nn.Linear(hidden_dim * heads2, hidden_dim) for _ in range(output_dim)
        ])

        # 输出层
        self.conv3 = GATConv(hidden_dim * heads2, output_dim, heads=1, concat=False)
        self.norm3 = BatchNorm(output_dim)

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        # 边dropout
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        # 基础特征提取
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 增加的中间处理层
        residual = x
        x = self.conv_mid(x, edge_index)
        x = self.norm_mid(x)
        x = F.elu(x)
        x = residual + x  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层处理
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 最终输出层
        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        return F.log_softmax(x, dim=1)