import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from gnn.layer import CausalRGCNConv, ImprovedCausalRGCNConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv

__all__ = [
    "UserBehaviorGCN",
    "GCN",
    "GAT",
    "GraphSAGE",
    "RGCN",
]

class UserBehaviorGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3, dropout=0.4, edge_dropout=0.05):
        # 可调整参数: num_relations, aggr
        super(UserBehaviorGCN, self).__init__()

        # 第一层 R-GCN
        # self.conv1 = CausalRGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations, aggr='max')
        self.conv1 = ImprovedCausalRGCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            aggr='max',  # 聚合方法：mean, add, max，目前性能好的组合：max + add / mean + add
            causal_strength=True,  # 启用因果强度预测
            sparsity=0.15  # 轻度稀疏性约束
        )
        self.ln1 = LayerNorm(hidden_dim)

        # 第二层 R-GCN
        # self.conv2 = CausalRGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations, aggr='add')
        self.conv2 = ImprovedCausalRGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            aggr='add',
            causal_strength=False,  # 不启用因果强度预测
            sparsity=0.15  # 轻度稀疏性约束
        )
        self.ln2 = LayerNorm(hidden_dim)

        # 分类头（输出层）：三层 MLP
        # self.classifier = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Sequential(
            # 第一层（保持维度）
            nn.Linear(hidden_dim, hidden_dim),  # 先保持原维度不变
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 使用高斯误差线性单元激活函数，比ReLU更平滑
            nn.Dropout(dropout),
            # 第二层（降维处理）
            nn.Linear(hidden_dim, hidden_dim // 2),  # 将特征维度减半
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.9),  # 逐层递减的 dropout 率
            # 输出层（进行分类）
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_type):
        # 在训练时对边进行 dropout, 增强模型泛化能力
        if self.training:
            # R-GCN 需要同时对 edge_index 和 edge_type 进行 dropout
            perm = torch.randperm(edge_index.size(1)) # 打乱边索引排列
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))] # 打乱后，从前向后保留指定比例的边
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]
            # if edge_attr is not None:
            #     edge_attr = edge_attr[keep_mask]

        # 第一层特征提取 - 使用边类型信息
        x = self.conv1(x, edge_index, edge_type)
        # x = self.conv1(x, edge_index, edge_type, edge_attr)
        x = self.ln1(x)
        x = F.gelu(x)
        x1 = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层特征提取 - 使用边类型信息
        x = self.conv2(x1, edge_index, edge_type)
        # x = self.conv2(x1, edge_index, edge_type, edge_attr)
        x = self.ln2(x)
        x = F.gelu(x)
        x = x + x1 * 0.5  # 轻度残差连接，保持特征流动性
        # 保留更完整的信息到分类层，不进行 dropout

        # 分类头进行分类
        x = self.classifier(x)  # 训练时间 ↑1.625倍（91->56），F1 ↑6.2%，acc ↑5%，pre ↑6%，recall ↑6.4%

        return F.log_softmax(x, dim=1)

    def get_node_reps(self, x, edge_index, edge_type):
        """
        获取节点的表示向量，用于计算Lp和Ln损失
        返回最终的节点特征表示（在分类前）

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            batch: 批处理索引（对于图级任务）

        返回:
            节点表示向量
        """
        if self.training:
            perm = torch.randperm(edge_index.size(1))
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))]
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]

        x = self.conv1(x, edge_index, edge_type)
        x = self.ln1(x)
        x = F.gelu(x)
        x1 = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x1, edge_index, edge_type)
        x = self.ln2(x)
        x = F.gelu(x)
        x = x + x1 * 0.5

        # 返回最终的节点表示
        return x

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

        self.conv1 = SAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggr='max'
        )
        self.ln1 = LayerNorm(hidden_dim)

        self.conv2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            aggr='add'
        )
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3, dropout=0.3, edge_dropout=0.1):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations, aggr='max')
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = RGCNConv(in_channels=hidden_dim, out_channels=output_dim, num_relations=num_relations, aggr='add')
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