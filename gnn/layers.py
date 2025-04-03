import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.utils import spmm
from torch_geometric.nn.conv import MessagePassing  # GNN消息传递基类
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import SparseTensor, torch_sparse

__all__ = [
    'CausalRGCNConv',
    'ImprovedCausalRGCNConv',
]

def masked_edge_index(edge_index, edge_mask):
    """
    根据边掩码筛选边索引

    Args:
        edge_index: 边索引，可以是普通张量(形状为[2, num_edges])或稀疏张量
        edge_mask: 布尔掩码，标识哪些边需要保留(True)或移除(False)

    Returns:
        筛选后的边索引，保持与输入相同的类型(Tensor或SparseTensor)
    """
    if isinstance(edge_index, Tensor):
        # 对于普通张量，直接使用布尔索引选择列
        return edge_index[:, edge_mask]
    # 对于SparseTensor，使用专用函数筛选非零元素
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class CausalRGCNConv(MessagePassing):
    """
    内存优化的因果关系图卷积网络层

    通过注意力机制动态调整不同关系类型的权重，增强模型对因果关系的建模能力。
    针对大规模图进行内存优化设计。
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases=None,
            aggr='mean',
            root_weight=True,
            bias=True,
            causal_attn=True,
            **kwargs
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.causal_attn = causal_attn

        # 处理输入通道
        self.in_channels_l = in_channels[0] if isinstance(in_channels, tuple) else in_channels

        # 创建权重矩阵
        if num_bases is not None and num_bases < num_relations:
            self.weight = Parameter(torch.empty(num_bases, self.in_channels_l, out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))
        else:
            self.weight = Parameter(torch.empty(num_relations, self.in_channels_l, out_channels))
            self.register_parameter('comp', None)

        # 自连接和偏置
        self.root = Parameter(torch.empty(self.in_channels_l, out_channels)) if root_weight else None
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        # 因果注意力机制参数 - 精简设计
        if causal_attn:
            # 关系类型重要性权重
            self.relation_attn = Parameter(torch.empty(num_relations, 1))
            # 节点特征投影矩阵 - 使用小维度减少计算开销
            self.attn_dim = min(8, self.in_channels_l)  # 降至较小维度
            self.attn_proj = Parameter(torch.empty(self.in_channels_l, self.attn_dim))
            self.attn_vec = Parameter(torch.empty(self.attn_dim * 2, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

        if self.causal_attn:
            glorot(self.relation_attn)
            if hasattr(self, 'attn_proj'):
                glorot(self.attn_proj)
                glorot(self.attn_vec)


    def forward(self, x, edge_index, edge_type=None):
        """
        前向传播函数 - 内存优化版本
        """
        # 处理输入特征
        x_l = x[0] if isinstance(x, tuple) else x
        x_r = x[1] if isinstance(x, tuple) else x_l

        # 处理离散特征
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        # 处理边类型
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # 初始化输出
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        size = (x_l.size(0), x_r.size(0))

        # 处理权重
        weight = self.weight
        if self.num_bases is not None and self.comp is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # 常规处理路径 - 分关系类型处理
        is_discrete = not torch.is_floating_point(x_l)

        # 预计算节点特征投影 - 只计算一次提高效率
        x_proj = None
        if self.causal_attn and torch.is_floating_point(x_l):
            x_proj = x_l @ self.attn_proj

        for i in range(self.num_relations):
            # 创建当前关系类型的边掩码
            mask = edge_type == i
            if not mask.any():
                continue

            # 获取当前关系的边子集
            tmp = masked_edge_index(edge_index, mask)

            # 计算边权重 - 针对当前关系类型的边子集计算
            edge_weight = None
            if self.causal_attn and isinstance(tmp, Tensor):
                row, col = tmp

                if torch.is_floating_point(x_l):
                    # 获取源节点和目标节点的投影特征
                    src_proj = x_proj[row]  # [num_edges, attn_dim]
                    dst_proj = x_proj[col]  # [num_edges, attn_dim]

                    # 拼接特征并计算注意力分数
                    alpha = torch.cat([src_proj, dst_proj], dim=-1) @ self.attn_vec  # [num_edges, 1]

                    # 添加关系特定权重 - 使用标量广播避免大张量
                    rel_weight = self.relation_attn[i].item()  # 获取标量值
                    alpha = alpha + rel_weight

                    # 应用sigmoid激活函数得到边权重
                    edge_weight = alpha.sigmoid().view(-1)

            # 传播消息
            if is_discrete:
                h = self.propagate(tmp, x=weight[i, x_l], edge_weight=edge_weight, size=size)
                out.add_(h)
            else:
                h = self.propagate(tmp, x=x_l, edge_weight=edge_weight, size=size)
                out.add_(h @ weight[i])

        # 处理自连接权重
        if self.root is not None:
            if not torch.is_floating_point(x_r):
                out.add_(self.root[x_r])
            else:
                out.add_(x_r @ self.root)

        # 添加偏置
        if self.bias is not None:
            out.add_(self.bias)

        return out

    def message(self, x_j, edge_weight=None):
        """定义消息函数"""
        # 如果有边权重，应用到消息上
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """组合消息生成和聚合步骤"""
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)


class ImprovedCausalRGCNConv(MessagePassing):
    """
    改进版因果关系图卷积网络层

    主要改进：
    1. 增加因果强度预测机制
    2. 采用双重注意力区分相关性和因果性
    3. 添加轻量级的干预支持
    4. 引入稀疏性约束以促进学习真实因果关系
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases=None,
            aggr='mean',
            root_weight=True,
            bias=True,
            causal_attn=True,
            causal_strength=True,  # 是否预测因果强度
            sparsity=0.15,  # 因果稀疏性控制
            **kwargs
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.causal_attn = causal_attn
        self.causal_strength = causal_strength
        self.sparsity = sparsity

        # 处理输入通道
        self.in_channels_l = in_channels[0] if isinstance(in_channels, tuple) else in_channels

        # 创建权重矩阵
        if num_bases is not None and num_bases < num_relations:
            self.weight = Parameter(torch.empty(num_bases, self.in_channels_l, out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))
        else:
            self.weight = Parameter(torch.empty(num_relations, self.in_channels_l, out_channels))
            self.register_parameter('comp', None)

        # 自连接和偏置
        self.root = Parameter(torch.empty(self.in_channels_l, out_channels)) if root_weight else None
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        # 因果注意力机制参数
        if causal_attn:
            # 关系类型重要性权重
            self.relation_attn = Parameter(torch.empty(num_relations, 1))

            # 节点特征投影维度 - 使用小维度减少计算开销
            self.attn_dim = min(8, self.in_channels_l)

            # 【改进1】：分离关联性和因果性的特征投影
            self.corr_proj = Parameter(torch.empty(self.in_channels_l, self.attn_dim))  # 关联性投影
            self.causal_proj = Parameter(torch.empty(self.in_channels_l, self.attn_dim))  # 因果性投影

            # 【改进2】：双重注意力向量
            self.corr_attn_vec = Parameter(torch.empty(self.attn_dim * 2, 1))  # 关联性注意力
            self.causal_attn_vec = Parameter(torch.empty(self.attn_dim * 2, 1))  # 因果性注意力

            # 【改进3】：关系特定的因果强度参数
            if causal_strength:
                self.causal_strength_weights = Parameter(torch.empty(num_relations, 1))
                self.causal_threshold = Parameter(torch.tensor(0.5))  # 可学习的因果阈值

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        super().reset_parameters()
        glorot(self.weight)
        if self.comp is not None:
            glorot(self.comp)
        if self.root is not None:
            glorot(self.root)
        if self.bias is not None:
            zeros(self.bias)

        if self.causal_attn:
            glorot(self.relation_attn)
            # 初始化投影矩阵和注意力向量
            glorot(self.corr_proj)
            glorot(self.causal_proj)
            glorot(self.corr_attn_vec)
            glorot(self.causal_attn_vec)

            # 初始化因果强度参数
            if self.causal_strength:
                # 初始化为较小值，避免过早引入太强的因果假设
                with torch.no_grad():
                    self.causal_strength_weights.data.fill_(0.1)
                    self.causal_threshold.data.fill_(0.5)

    def forward(self, x, edge_index, edge_type=None, intervention_nodes=None):
        """
        前向传播函数 - 支持基本的因果干预

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            intervention_nodes: 可选，要进行干预的节点索引
        """
        # 处理输入特征
        x_l = x[0] if isinstance(x, tuple) else x
        x_r = x[1] if isinstance(x, tuple) else x_l

        # 处理离散特征
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        # 处理边类型
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # 【改进4】：支持干预
        if intervention_nodes is not None:
            # 创建干预掩码，移除涉及干预节点的边
            if isinstance(edge_index, Tensor):
                src, dst = edge_index
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
                for node in intervention_nodes:
                    # 移除所有与干预节点相关的边
                    node_mask = (src == node) | (dst == node)
                    edge_mask &= ~node_mask

                # 应用掩码
                edge_index = edge_index[:, edge_mask]
                edge_type = edge_type[edge_mask]

        # 初始化输出
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        size = (x_l.size(0), x_r.size(0))

        # 处理权重
        weight = self.weight
        if self.num_bases is not None and self.comp is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # 常规处理路径 - 分关系类型处理
        is_discrete = not torch.is_floating_point(x_l)

        # 预计算节点特征投影 - 只计算一次提高效率
        x_corr_proj, x_causal_proj = None, None
        if self.causal_attn and torch.is_floating_point(x_l):
            x_corr_proj = x_l @ self.corr_proj  # 关联性投影
            x_causal_proj = x_l @ self.causal_proj  # 因果性投影

        # 保存每种关系的因果强度分数
        causal_scores = []

        for i in range(self.num_relations):
            # 创建当前关系类型的边掩码
            mask = edge_type == i
            if not mask.any():
                continue

            # 获取当前关系的边子集
            tmp = masked_edge_index(edge_index, mask)

            # 计算边权重 - 针对当前关系类型的边子集计算
            edge_weight = None
            causal_score = None

            if self.causal_attn and isinstance(tmp, Tensor):
                row, col = tmp

                if torch.is_floating_point(x_l):
                    # 计算关联性注意力
                    src_corr = x_corr_proj[row]  # [num_edges, attn_dim]
                    dst_corr = x_corr_proj[col]  # [num_edges, attn_dim]
                    corr_attn = torch.cat([src_corr, dst_corr], dim=-1) @ self.corr_attn_vec

                    # 计算因果性注意力
                    src_causal = x_causal_proj[row]  # [num_edges, attn_dim]
                    dst_causal = x_causal_proj[col]  # [num_edges, attn_dim]
                    causal_attn = torch.cat([src_causal, dst_causal], dim=-1) @ self.causal_attn_vec

                    # 添加关系特定权重
                    rel_weight = self.relation_attn[i]

                    # 【改进5】：组合关联性和因果性注意力
                    if self.causal_strength:
                        # 计算因果强度
                        causal_factor = torch.sigmoid(self.causal_strength_weights[i])
                        # 加权组合：关联性和因果性
                        alpha = (1 - causal_factor) * corr_attn + causal_factor * causal_attn + rel_weight
                        # 记录因果分数以便后续分析
                        causal_score = causal_factor * torch.sigmoid(causal_attn)
                    else:
                        # 简单加权求和
                        alpha = 0.5 * (corr_attn + causal_attn) + rel_weight

                    # 应用sparsity约束 - 鼓励稀疏的因果关系
                    if self.training and self.sparsity > 0:
                        # 随机丢弃一部分边，促进稀疏性
                        mask = torch.rand_like(alpha) > self.sparsity
                        alpha = alpha * mask

                    # 应用sigmoid激活函数得到边权重
                    edge_weight = torch.sigmoid(alpha).view(-1)

            # 传播消息
            if is_discrete:
                h = self.propagate(tmp, x=weight[i, x_l], edge_weight=edge_weight, size=size)
                out.add_(h)
            else:
                h = self.propagate(tmp, x=x_l, edge_weight=edge_weight, size=size)
                out.add_(h @ weight[i])

            # 保存因果分数
            if causal_score is not None:
                causal_scores.append((i, causal_score))

        # 处理自连接权重
        if self.root is not None:
            if not torch.is_floating_point(x_r):
                out.add_(self.root[x_r])
            else:
                out.add_(x_r @ self.root)

        # 添加偏置
        if self.bias is not None:
            out.add_(self.bias)

        # 返回输出结果，如果需要可以同时返回因果分数
        return out

    def message(self, x_j, edge_weight=None):
        """定义消息函数"""
        # 如果有边权重，应用到消息上
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """组合消息生成和聚合步骤"""
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def do_intervention(self, x, edge_index, edge_type, target_nodes):
        """
        执行因果干预，移除与目标节点相关的边，分析对结果的影响

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            target_nodes: 要干预的目标节点列表

        返回:
            干预结果和原始结果的差异
        """
        # 原始预测
        original = self.forward(x, edge_index, edge_type)

        # 干预后预测
        intervened = self.forward(x, edge_index, edge_type, intervention_nodes=target_nodes)

        # 计算干预效应
        effect = original - intervened

        return {
            'original': original,
            'intervened': intervened,
            'effect': effect
        }

    def causal_regularization_loss(self):
        """
        计算因果稀疏性正则化损失，鼓励学习稀疏的因果关系

        返回:
            正则化损失
        """
        if not self.causal_strength:
            return 0.0

        # L1正则化促进稀疏性
        l1_reg = torch.norm(self.causal_strength_weights, p=1)

        return self.sparsity * l1_reg