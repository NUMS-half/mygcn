import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv, GATConv

# 1. Base GCNConv - 2 layers
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.ln1 = LayerNorm(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#         self.ln2 = LayerNorm(output_dim)
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#
#     def forward(self, x, edge_index):
#         # 对边进行 dropout
#         edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = self.ln1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.ln2(x)
#         return F.log_softmax(x, dim=1)

# 2. Base GCNConv - 3 layers
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 新增一层
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)


# # 3. GATConv - 2 layers
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, heads=3, dropout=0.5, edge_dropout=0.2):
#         super(GCN, self).__init__()
#         # 第一个GAT卷积层
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
#         # 第一层的LayerNorm，注意维度是hidden_dim * heads因为输出是拼接的
#         self.norm1 = LayerNorm(hidden_dim * heads)
#
#         # 第二个GAT卷积层
#         self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
#         # 第二层的LayerNorm
#         self.norm2 = LayerNorm(output_dim)
#
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         # 第一层: 卷积 -> 归一化 -> 激活 -> dropout
#         x = self.conv1(x, edge_index)
#         x = self.norm1(x)  # 添加的LayerNorm
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 第二层: 卷积 -> 归一化 -> softmax
#         x = self.conv2(x, edge_index)
#         x = self.norm2(x)  # 添加的LayerNorm
#
#         return F.log_softmax(x, dim=1)

# # 4. GATConv - 3 layers
class StableGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads1=6, heads2=3, dropout=0.5, edge_dropout=0.2):
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



# class StableGAT(torch.nn.Module):
#     def __init__(self,
#                  input_dim,
#                  hidden_dim,
#                  output_dim,
#                  heads1=4,
#                  heads2=2,
#                  dropout=0.5,
#                  attn_dropout=0.3,
#                  edge_dropout=0.2,
#                  use_residual=True):
#         super(StableGAT, self).__init__()
#
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#         self.use_residual = use_residual
#
#         # 第1层GAT
#         self.conv1 = GATConv(
#             input_dim, hidden_dim,
#             heads=heads1,
#             concat=True,
#             dropout=attn_dropout,
#             add_self_loops=True  # 关键修改点：恢复默认自环
#         )
#         self.norm1 = LayerNorm(hidden_dim * heads1)
#
#         # 第2层GAT
#         self.conv2 = GATConv(
#             hidden_dim * heads1, hidden_dim,
#             heads=heads2,
#             concat=True,
#             dropout=attn_dropout,
#             add_self_loops=True  # 关键修改点
#         )
#         self.norm2 = LayerNorm(hidden_dim * heads2)
#
#         # 输出层
#         self.conv3 = GATConv(
#             hidden_dim * heads2, output_dim,
#             heads=1,
#             concat=False,
#             dropout=attn_dropout
#         )
#         self.norm3 = LayerNorm(output_dim)
#
#         # 残差连接投影层
#         if self.use_residual:
#             self.proj1 = nn.Linear(input_dim, hidden_dim * heads1)
#             self.proj2 = nn.Linear(hidden_dim * heads1, hidden_dim * heads2)
#
#         # 更安全的参数初始化
#         self._init_weights()
#
#     def _init_weights(self):
#         """ 兼容不同PyG版本的初始化 """
#         for m in self.modules():
#             if isinstance(m, GATConv):
#                 # 检查线性层是否存在
#                 if hasattr(m, 'lin') and m.lin is not None:  # 兼容旧版本
#                     nn.init.xavier_normal_(m.lin.weight)
#                 elif hasattr(m, 'lin_src') and m.lin_src is not None:  # 新版本
#                     nn.init.xavier_normal_(m.lin_src.weight)
#                     if m.lin_dst is not None:
#                         nn.init.xavier_normal_(m.lin_dst.weight)
#
#                 # 初始化注意力参数
#                 if hasattr(m, 'att_src'):
#                     nn.init.xavier_uniform_(m.att_src)
#                 if hasattr(m, 'att_dst') and m.att_dst is not None:
#                     nn.init.xavier_uniform_(m.att_dst)
#
#                 # 初始化偏置项
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x, edge_index):
#         x_res = x if self.use_residual else None
#
#         if self.edge_dropout > 0 and self.training:
#             edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout)
#
#         # 第1层
#         x = self.conv1(x, edge_index)
#         x = self.norm1(x)
#         if self.use_residual:
#             x += self.proj1(x_res) if x_res.shape != x.shape else x_res
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 第2层
#         x_res = x if self.use_residual else None
#         x = self.conv2(x, edge_index)
#         x = self.norm2(x)
#         if self.use_residual:
#             x += self.proj2(x_res) if x_res.shape != x.shape else x_res
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 输出层
#         x = self.conv3(x, edge_index)
#         x = self.norm3(x)
#
#         return F.log_softmax(x, dim=1)
