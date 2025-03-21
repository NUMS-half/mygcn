import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm


class SemanticAttention(nn.Module):
    """语义级注意力层，用于聚合不同元路径的表示"""

    def __init__(self, in_dim, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        """z: [num_relations, num_nodes, hidden_dim]"""
        w = self.project(z)  # [num_relations, num_nodes, 1]
        beta = torch.softmax(w, dim=0)  # 对关系类型做softmax
        return (beta * z).sum(0)  # [num_nodes, hidden_dim]


class HANLayer(nn.Module):
    """HAN层：整合节点级和语义级注意力"""

    def __init__(self, hidden_dim, num_relations, heads=8, dropout=0.3):
        super(HANLayer, self).__init__()

        # 为每种关系类型创建一个GAT卷积
        self.gat_layers = nn.ModuleList()
        for i in range(num_relations):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # 语义级注意力层
        self.semantic_attention = SemanticAttention(in_dim=hidden_dim, hidden_dim=256)
        self.num_relations = num_relations

    def forward(self, x, edge_indices, edge_types):
        """
        x: 节点特征 [num_nodes, hidden_dim]
        edge_indices: 所有边的索引 [2, num_edges]
        edge_types: 每条边的类型 [num_edges]
        """
        semantic_embeddings = []

        # 对每种关系分别应用GAT
        for rel_type in range(self.num_relations):
            # 提取该关系类型的边
            mask = (edge_types == rel_type)
            if mask.sum() > 0:  # 确保有这种类型的边
                rel_edge_index = edge_indices[:, mask]

                # 应用GAT
                rel_embedding = self.gat_layers[rel_type](x, rel_edge_index)
                semantic_embeddings.append(rel_embedding)

        # 如果某些关系类型没有边，填充零张量
        while len(semantic_embeddings) < self.num_relations:
            semantic_embeddings.append(torch.zeros_like(x))

        # 语义级注意力聚合
        semantic_embeddings = torch.stack(semantic_embeddings, dim=0)  # [num_relations, num_nodes, hidden_dim]
        return self.semantic_attention(semantic_embeddings)  # [num_nodes, hidden_dim]


class HAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=3, dropout=0.35, edge_dropout=0.1):
        super(HAN, self).__init__()

        # 特征转换层：增强输入特征表示
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 多层HAN架构：替代原来的RGCN层
        self.han_layer1 = HANLayer(hidden_dim, num_relations, heads=8, dropout=dropout)
        self.ln1 = LayerNorm(hidden_dim)

        self.han_layer2 = HANLayer(hidden_dim, num_relations, heads=4, dropout=dropout)
        self.ln2 = LayerNorm(hidden_dim)

        self.han_layer3 = HANLayer(hidden_dim, num_relations, heads=4, dropout=dropout)
        self.ln3 = LayerNorm(hidden_dim)

        # 多头注意力增强特征：在不同语义空间上聚合信息
        self.global_attention = GATConv(
            hidden_dim,
            hidden_dim // 4,
            heads=4,
            dropout=dropout
        )
        self.ln_att = LayerNorm(hidden_dim)

        # 特征融合层：整合不同层次特征
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 关系重要性学习
        self.relation_importance = nn.Parameter(torch.ones(num_relations))

        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.num_relations = num_relations

    def forward(self, x, edge_index, edge_type):
        # 边缘dropout（训练阶段）
        if self.training and self.edge_dropout > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        # 初始特征转换
        x_0 = self.feature_encoder(x)

        # 多层次特征抽取，使用HAN层替代RGCN
        x_1 = self.han_layer1(x_0, edge_index, edge_type)
        x_1 = F.relu(self.ln1(x_1))
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_2 = self.han_layer2(x_1, edge_index, edge_type)
        x_2 = F.relu(self.ln2(x_2))
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)

        # 带有残差连接的第三层
        x_3 = self.han_layer3(x_2, edge_index, edge_type)
        x_3 = F.relu(self.ln3(x_3))
        x_3 = x_3 + x_1 * 0.2 + x_2 * 0.3  # 多层残差连接
        x_3 = F.dropout(x_3, p=self.dropout, training=self.training)

        # 全局注意力增强
        x_att = self.global_attention(x_3, edge_index)
        x_att = self.ln_att(x_att)
        x_att = F.dropout(x_att, p=self.dropout, training=self.training)

        # 多层特征融合
        multi_scale = torch.cat([x_1, x_2, x_3], dim=-1)
        fused_features = self.fusion(multi_scale)

        # 结合注意力特征和融合特征
        final_features = torch.cat([fused_features, x_att], dim=-1)

        # 分类
        logits = self.classifier(final_features)

        return F.log_softmax(logits, dim=1)

    def get_node_reps(self, x, edge_index, edge_type, batch=None):
        """获取节点表示向量，用于计算损失或可视化"""
        # 边缘dropout（训练阶段）
        if self.training and self.edge_dropout > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        # 初始特征转换
        x_0 = self.feature_encoder(x)

        # 多层次特征抽取
        x_1 = self.han_layer1(x_0, edge_index, edge_type)
        x_1 = F.relu(self.ln1(x_1))
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_2 = self.han_layer2(x_1, edge_index, edge_type)
        x_2 = F.relu(self.ln2(x_2))
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)

        x_3 = self.han_layer3(x_2, edge_index, edge_type)
        x_3 = F.relu(self.ln3(x_3))
        x_3 = x_3 + x_1 * 0.2 + x_2 * 0.3  # 残差连接
        x_3 = F.dropout(x_3, p=self.dropout, training=self.training)

        # 多层特征融合
        multi_scale = torch.cat([x_1, x_2, x_3], dim=-1)
        fused_features = self.fusion(multi_scale)

        return fused_features