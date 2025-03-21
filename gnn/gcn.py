import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, GATConv


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

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1):
#         super(GCN, self).__init__()
#         # 扩展隐藏层维度以增加模型容量
#         mid_dim = hidden_dim * 2
#
#         # 第一层卷积
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.ln1 = nn.LayerNorm(hidden_dim)
#
#         # 添加中间层增强特征提取能力
#         self.conv_mid = GCNConv(hidden_dim, mid_dim)
#         self.ln_mid = nn.LayerNorm(mid_dim)
#
#         # 第二层卷积 (保持原有连接)
#         self.conv2 = GCNConv(mid_dim, hidden_dim)
#         self.ln2 = nn.LayerNorm(hidden_dim)
#
#         # 添加类别感知层 - 针对多分类问题
#         self.class_attention = nn.Linear(hidden_dim, hidden_dim)
#
#         # 输出分类层
#         self.classifier = nn.Linear(hidden_dim, output_dim)
#         self.ln_out = nn.LayerNorm(output_dim)
#
#         # 修正残差连接投影维度
#         self.shortcut1 = nn.Linear(input_dim, mid_dim)
#         # 修改: 从hidden_dim → hidden_dim (不是从mid_dim → hidden_dim)
#         self.shortcut2 = nn.Linear(hidden_dim, hidden_dim)
#
#         # Dropout参数
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#
#     def forward(self, x, edge_index):
#         # 原始输入保存用于残差连接
#         identity = x
#
#         # 对边进行 dropout (仅在训练时)
#         if self.training:
#             edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout)
#
#         # 第一层: 基础特征提取
#         x = self.conv1(x, edge_index)
#         x = self.ln1(x)
#         x = F.gelu(x)  # 使用GELU激活改进非线性表达能力
#
#         # 保存中间结果用于后续残差连接 (形状是[batch_size, hidden_dim])
#         mid_identity = x
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 中间层: 增强特征表达
#         x = self.conv_mid(x, edge_index)
#         x = self.ln_mid(x)
#
#         # 添加第一个残差连接 (需要投影维度)
#         x = x + self.shortcut1(identity)
#         x = F.gelu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 第二层: 特征整合
#         x = self.conv2(x, edge_index)
#         x = self.ln2(x)
#
#         # 添加第二个残差连接 - 这里mid_identity形状是[batch_size, hidden_dim]
#         x = x + self.shortcut2(mid_identity)  # 现在形状匹配了
#         x = F.gelu(x)
#
#         # 类别感知机制: 增强多分类能力
#         attention = torch.sigmoid(self.class_attention(x))
#         x = x * attention
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 最终分类层
#         x = self.classifier(x)
#         x = self.ln_out(x)
#
#         return F.log_softmax(x, dim=1)
#
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, edge_dropout=0.1):
#         super(GCN, self).__init__()
#         # 保持原始结构但增强处理能力
#         self.conv1 = GCNConv(input_dim, hidden_dim, improved=True, add_self_loops=True)
#         self.ln1 = LayerNorm(hidden_dim)
#
#         # 简单而有效的第二层
#         self.conv2 = GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
#         self.ln2 = LayerNorm(hidden_dim)
#
#         self.classifier = nn.Linear(hidden_dim, output_dim)
#
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#
#     def forward(self, x, edge_index):
#         # 对边进行 dropout (仅在训练时)
#         edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
#
#         # 第一层特征提取
#         x = self.conv1(x, edge_index)
#         x = self.ln1(x)
#         x = F.relu(x)
#         x1 = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 第二层特征提取
#         x = self.conv2(x1, edge_index)
#         x = self.ln2(x)
#         x = F.relu(x)
#         x = x + x1 * 0.2  # 轻量级残差连接，保持特征流动性
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # 分类
#         x = self.classifier(x)
#
#         return F.log_softmax(x, dim=1)


class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3, dropout=0.3, edge_dropout=0.1):
        super(RGCN, self).__init__()
        # num_relations：表示图中不同关系的数量

        # 第一层 R-GCN
        self.conv1 = RGCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            # R-GCN 会默认添加自环
            aggr='add'  # 聚合方法：mean, add, max
        )
        self.ln1 = LayerNorm(hidden_dim)

        # 第二层 R-GCN
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            aggr='max'
        )
        self.ln2 = LayerNorm(hidden_dim)

        # 分类头：三层 MLP
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
            nn.Dropout(dropout * 0.8),  # 逐层递减的 dropout 率
            # 输出层（进行分类）
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_type):
        # 在训练时对边进行 dropout
        if self.training:
            # R-GCN 需要同时对 edge_index 和 edge_type 进行 dropout
            perm = torch.randperm(edge_index.size(1)) # 打乱边索引排列
            keep_mask = perm[:int(edge_index.size(1) * (1 - self.edge_dropout))] # 打乱后，从前向后保留指定比例的边
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
        x = x + x1 * 0.2  # 轻度残差连接，保持特征流动性
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 分类头进行分类
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

# class RGCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3, dropout=0.3, edge_dropout=0.1):
#         super(RGCN, self).__init__()
#
#         # 特征转换层
#         self.feature_encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.8)
#         )
#
#         # RGCN卷积层 - 使用不同的聚合方式
#         self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, aggr='add')
#         self.ln1 = LayerNorm(hidden_dim)
#
#         self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, aggr='mean')
#         self.ln2 = LayerNorm(hidden_dim)
#
#         self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations, aggr='max')
#         self.ln3 = LayerNorm(hidden_dim)
#
#         # 多头注意力层
#         self.attention = GATConv(
#             hidden_dim,
#             hidden_dim // 4,
#             heads=4,
#             dropout=dropout
#         )
#         self.ln_att = LayerNorm(hidden_dim)
#
#         # 特征融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.7)
#         )
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.8),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.6),
#             nn.Linear(hidden_dim // 2, output_dim)
#         )
#
#         # 模型参数
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#         self.num_relations = num_relations
#
#     def _preprocess_edges(self, edge_index, edge_type):
#         """边预处理函数：应用边dropout"""
#         if self.training and self.edge_dropout > 0:
#             mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
#             edge_index = edge_index[:, mask]
#             edge_type = edge_type[mask]
#         return edge_index, edge_type
#
#     def _extract_features(self, x, edge_index, edge_type):
#         """特征提取核心功能：返回多个层次的特征"""
#         # 初始特征转换
#         x_0 = self.feature_encoder(x)
#
#         # 多层次特征抽取
#         x_1 = F.relu(self.ln1(self.conv1(x_0, edge_index, edge_type)))
#         x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
#
#         x_2 = F.relu(self.ln2(self.conv2(x_1, edge_index, edge_type)))
#         x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
#
#         # 带有残差连接的第三层
#         x_3 = F.relu(self.ln3(self.conv3(x_2, edge_index, edge_type)))
#         x_3 = x_3 + x_1 * 0.2 + x_2 * 0.3  # 多层残差连接
#         x_3 = F.dropout(x_3, p=self.dropout, training=self.training)
#
#         # 注意力层增强特征学习
#         x_att = self.attention(x_3, edge_index)
#         x_att = self.ln_att(x_att)
#         x_att = F.dropout(x_att, p=self.dropout, training=self.training)
#
#         # 多层特征融合
#         multi_scale = torch.cat([x_1, x_2, x_3], dim=-1)
#         fused_features = self.fusion(multi_scale)
#
#         return fused_features, x_att
#
#     def forward(self, x, edge_index, edge_type):
#         # 边缘预处理
#         edge_index, edge_type = self._preprocess_edges(edge_index, edge_type)
#
#         # 提取特征
#         fused_features, x_att = self._extract_features(x, edge_index, edge_type)
#
#         # 结合注意力特征和融合特征
#         final_features = torch.cat([fused_features, x_att], dim=-1)
#
#         # 分类
#         logits = self.classifier(final_features)
#
#         return F.log_softmax(logits, dim=1)
#
#     def get_node_reps(self, x, edge_index, edge_type, batch=None):
#         """获取节点的表示向量，用于计算Lp和Ln损失"""
#         # 边缘预处理
#         edge_index, edge_type = self._preprocess_edges(edge_index, edge_type)
#
#         # 提取特征并返回融合特征
#         fused_features, _ = self._extract_features(x, edge_index, edge_type)
#
#         return fused_features