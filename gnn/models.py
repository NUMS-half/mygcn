import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from gnn.layers import CausalRGCNConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv

__all__ = [
    "UserBehaviorGCN",
    "GCN",
    "GAT",
    "GraphSAGE",
    "RGCN",
    "CausalRGCN"
]


class UserBehaviorGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=9, dropout=0.3, edge_dropout=0.05):
        # 可调整参数: num_relations, aggr
        super(UserBehaviorGCN, self).__init__()

        # 第一层 R-GCN
        self.conv1 = CausalRGCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=6,
            aggr='max',  # 聚合方法：mean, add, max，目前性能好的组合：max + add / mean + add
            causal_strength=True,  # 启用因果强度预测
            sparsity=0.15,  # 轻度稀疏性约束
        )
        self.ln1 = LayerNorm(hidden_dim)

        # 第二层 R-GCN
        self.conv2 = CausalRGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=6,
            aggr='add',
            causal_strength=True,
            sparsity=0.25,  # 稀疏性约束递增
        )
        self.ln2 = LayerNorm(hidden_dim)

        # 分类头（输出层）：三层 MLP
        self.classifier = nn.Sequential(
            # 第一层（保持维度）
            nn.Linear(hidden_dim, hidden_dim),  # 先保持原维度不变
            nn.GELU(),
            nn.Dropout(dropout),
            # 第二层（降维处理）
            nn.Linear(hidden_dim, hidden_dim // 2),  # 将特征维度减半
            nn.GELU(),
            nn.Dropout(dropout * 0.9),  # 逐层递减的 dropout 率
            # 输出层（进行分类）
            nn.Linear(hidden_dim // 2, output_dim)
        )
        # self.classifier = nn.Linear(hidden_dim, output_dim)

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_type):
        # 在训练时对边进行 dropout, 增强模型泛化能力
        if self.training:
            # 注意：R-GCN 需要同时对 edge_index 和 edge_type 进行 dropout
            perm = torch.randperm(edge_index.size(1))  # 打乱边索引排列
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))]  # 打乱后，从前向后保留指定比例的边
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]

        # 第一层特征提取 - 使用边类型信息
        x = self.conv1(x, edge_index, edge_type)
        x = self.ln1(x)
        x = F.relu(x)
        x1 = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层特征提取 - 使用边类型信息
        x = self.conv2(x1, edge_index, edge_type)
        x = self.ln2(x)
        x = F.relu(x)
        x = x + x1 * 0.5  # 轻度残差连接，保持特征流动性

        # 分类头进行分类
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    def get_node_reps(self, x, edge_index, edge_type):
        """
        按原论文方式实现节点表示获取函数

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型

        返回:
            node_x: 节点表示向量
        """
        # 使用第一层卷积直接作为嵌入层，保持模型结构
        convs = [self.conv1, self.conv2]
        activations = [F.relu, F.relu]
        norms = [self.ln1, self.ln2]

        # 初始处理 - 在训练时对边进行dropout
        if self.training:
            perm = torch.randperm(edge_index.size(1))
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))]
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]

        # 按原论文的循环处理
        x_prev = None
        for i, (conv, activation, norm) in enumerate(zip(convs, activations, norms)):
            # 卷积层
            x_new = conv(x=x, edge_index=edge_index, edge_type=edge_type)
            # 归一化层
            x_new = norm(x_new)
            # 激活函数
            x_new = activation(x_new)

            # 存储第一层输出用于残差连接
            if i == 0:
                x_prev = x_new
                # 仅在第一层后应用dropout，与原模型一致
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            x = x_new

        # 添加轻度残差连接，保持原模型的特性
        x = x + x_prev * 0.5

        # 返回最终的节点表示
        node_x = x
        return node_x


# 1. GCN - 2 GCNConv layers
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.ln2 = LayerNorm(output_dim)
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        return F.log_softmax(x, dim=1)


# 2. GAT - 2 GATConv layers
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1, heads=8):
        super(GAT, self).__init__()

        # 第一层GAT
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // heads,  # 将hidden_dim分成heads份
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.ln1 = LayerNorm(hidden_dim)

        # 第二层GAT
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.ln2 = LayerNorm(output_dim)

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.ln2(x)

        return F.log_softmax(x, dim=1)


# 3. GraphSAGE - 2 SAGEConv layers
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1):
        # 可调整参数: aggr
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, aggr='max')
        self.ln1 = LayerNorm(hidden_dim)

        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim, aggr='add')
        self.ln2 = LayerNorm(output_dim)

        # 参数
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.ln2(x)

        return F.log_softmax(x, dim=1)


# 4. R-GCN - 2 RGCNConv layers
class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=9, dropout=0.3, edge_dropout=0.1):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases=9,
                              aggr='max')
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = RGCNConv(in_channels=hidden_dim, out_channels=output_dim, num_relations=num_relations, num_bases=9,
                              aggr='add')
        self.ln2 = LayerNorm(output_dim)
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_type):
        if self.training:
            perm = torch.randperm(edge_index.size(1))
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))]
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]

        x = self.conv1(x, edge_index, edge_type)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = self.ln2(x)

        return F.log_softmax(x, dim=1)


# 5. CausalRGCN - 2 CausalRGCNConv layers
class CausalRGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=9, dropout=0.3, edge_dropout=0.1):
        super(CausalRGCN, self).__init__()
        self.conv1 = CausalRGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations,
                                    num_bases=9, aggr='max', causal_strength=True, sparsity=0.15)
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = CausalRGCNConv(in_channels=hidden_dim, out_channels=output_dim,
                                    num_relations=num_relations,
                                    num_bases=9, aggr='add', causal_strength=True, sparsity=0.25)
        self.ln2 = LayerNorm(output_dim)
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_type):
        if self.training:
            perm = torch.randperm(edge_index.size(1))
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))]
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]

        x = self.conv1(x, edge_index, edge_type)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = self.ln2(x)

        return F.log_softmax(x, dim=1)
