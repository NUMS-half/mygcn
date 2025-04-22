import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.utils import spmm
from torch_geometric.nn.conv import MessagePassing  # GNN消息传递基类
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import SparseTensor, torch_sparse


def masked_edge_index(edge_index, edge_mask):
    """
    根据边掩码筛选边索引

    参数:
        edge_index: 边索引，格式为[2, num_edges]的张量或SparseTensor
        edge_mask: 布尔掩码，标记需要保留的边(True)

    返回:
        经过掩码筛选后的边索引
    """
    if isinstance(edge_index, Tensor):
        # 对于普通张量，直接使用布尔索引选择列
        return edge_index[:, edge_mask]
    # 对于SparseTensor，使用PyG专用函数筛选非零元素
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class CausalRGCNConv(MessagePassing):
    """
    基于因果注意力的关系图卷积网络层

    基于PyG的RGCNConv，增加了以下特性以增强因果推理能力:
    1. 双重注意力机制 - 区分节点间的相关性和因果性
    2. 因果强度预测 - 量化边的因果影响程度
    3. 干预支持 - 实现因果推断中的do-calculus操作
    4. 稀疏性约束 - 促进真实因果结构的学习
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases=None,
            aggr='mean',
            root_weight=True,  # 默认包含自连接权重（与RGCN相同）
            bias=True,
            causal_attn=True,  # 新增：是否使用因果注意力机制
            causal_strength=True,  # 新增：是否预测因果强度
            sparsity=0.15,  # 新增：控制因果关系的稀疏程度
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

        # 1. 确定输入通道
        self.in_channels_l = in_channels[0] if isinstance(in_channels, tuple) else in_channels

        # 2. 创建关系特定权重矩阵
        if num_bases is not None and num_bases < num_relations:
            self.weight = Parameter(torch.empty(num_bases, self.in_channels_l, out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))  # 关系组合系数参数
        else:
            self.weight = Parameter(torch.empty(num_relations, self.in_channels_l, out_channels))
            self.register_parameter('comp', None)

        # 3. 定义自连接权重和偏置
        self.root = Parameter(torch.empty(self.in_channels_l, out_channels)) if root_weight else None
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        # 4. [新增] 因果注意力机制
        if causal_attn:
            # 关系权重：为每种关系分配重要性权重
            self.relation_attn = Parameter(torch.empty(num_relations, 1))

            # 设置注意力特征投影维度（这里使用较小维度减少计算负担）
            self.attn_dim = min(8, self.in_channels_l)

            # 区分关联性和因果性
            # 双重特征投影：将节点特征投影到设定的维度
            self.corr_proj = Parameter(torch.empty(self.in_channels_l, self.attn_dim))  # 关联性投影
            self.causal_proj = Parameter(torch.empty(self.in_channels_l, self.attn_dim))  # 因果性投影

            # 双重注意力向量：用于计算节点之间的"统计关联性"和"因果关系强度"
            self.corr_attn_vec = Parameter(torch.empty(self.attn_dim * 2, 1))  # 关联性注意力（乘2是因为连接源节点与目标节点的特征）
            self.causal_attn_vec = Parameter(torch.empty(self.attn_dim * 2, 1))  # 因果性注意力

            # 5. [新增] 因果强度控制
            if causal_strength:
                # 每种关系的因果强度权重
                self.causal_strength_weights = Parameter(torch.empty(num_relations, 1))

        # 初始化所有参数
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化函数"""
        # 初始化权重和偏置等原始参数（Glorot 初始化）
        super().reset_parameters()
        glorot(self.weight)
        if self.comp is not None:
            glorot(self.comp)
        if self.root is not None:
            glorot(self.root)
        if self.bias is not None:
            zeros(self.bias)  # 偏置初始化为0

        # 新增：初始化因果注意力相关参数
        if self.causal_attn:
            glorot(self.relation_attn)
            glorot(self.corr_proj)
            glorot(self.causal_proj)
            glorot(self.corr_attn_vec)
            glorot(self.causal_attn_vec)

            # 初始化因果强度参数
            if self.causal_strength:
                # 较小的初始值，避免过早引入强因果假设
                with torch.no_grad():
                    self.causal_strength_weights.data.fill_(0.1)  # 将因果强度权重初始化为较小值，避免模型过早强制建立因果关系

    def forward(self, x, edge_index, edge_type=None, intervention_nodes=None):
        """
        前向传播函数，支持因果干预操作

        参数:
            x: 节点特征矩阵
            edge_index: 边索引
            edge_type: 关系类型
            intervention_nodes: [新增参数] 指定要进行干预的节点索引

        返回:
            更新后的节点表示
        """
        # 1. 处理输入特征
        x_l = x[0] if isinstance(x, tuple) else x
        x_r = x[1] if isinstance(x, tuple) else x_l

        # 2. 处理离散特征
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        # 3. 边类型提取
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None  # 确保边类型不为空

        # 4. [新增] 干预支持
        if intervention_nodes is not None:
            # 实现do-calculus，移除与干预节点相关的所有边
            if isinstance(edge_index, Tensor):
                # 分离源节点和目标节点
                src, dst = edge_index
                # 创建初始掩码，默认保留所有边
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
                # 通过掩码进行边筛选操作
                for node in intervention_nodes:
                    # 识别与干预节点相关的所有边（入边和出边）
                    node_mask = (src == node) | (dst == node)
                    # 从当前掩码中移除这些边（逻辑与非操作）
                    edge_mask &= ~node_mask

                # 应用掩码
                edge_index = edge_index[:, edge_mask]
                edge_type = edge_type[edge_mask]

        # 5. 初始化输出张量
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        size = (x_l.size(0), x_r.size(0))

        # 6. 处理关系权重
        weight = self.weight
        if self.num_bases is not None and self.comp is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # 7. 判断是否为离散特征
        is_discrete = not torch.is_floating_point(x_l)

        # 8. [新增] 定义并计算所有节点的特征投影，提高计算效率
        x_corr_proj, x_causal_proj = None, None
        if self.causal_attn and torch.is_floating_point(x_l):
            x_corr_proj = x_l @ self.corr_proj  # 关联性投影
            x_causal_proj = x_l @ self.causal_proj  # 因果性投影

        # 9. [新增] 存储各关系类型的因果强度得分
        causal_scores = []

        # 10. 开始依次处理各关系类型
        for i in range(self.num_relations):
            # 1) 获取当前关系类型的边掩码
            mask = edge_type == i
            if not mask.any():
                continue

            # 2) 获取当前类型关系的边(子)集
            sub_edges = masked_edge_index(edge_index, mask)

            # 3) [新增] 定义边权重与因果分数
            edge_weight = None
            causal_score = None

            # 4) 双重注意力计算
            if self.causal_attn and isinstance(sub_edges, Tensor):
                src, dst = sub_edges  # 获取包含每条边的源节点、目标节点索引

                if torch.is_floating_point(x_l):
                    # 计算关联性注意力
                    src_corr = x_corr_proj[src]  # [num_edges, attn_dim]: 每行对应一条边的源节点特征
                    dst_corr = x_corr_proj[dst]  # [num_edges, attn_dim]
                    # 先沿特征维度拼接源节点和目标节点特征（叠加），然后与注意力向量相乘，得到注意力分数
                    corr_attn = torch.cat([src_corr, dst_corr], dim=-1) @ self.corr_attn_vec

                    # 计算因果性注意力
                    src_causal = x_causal_proj[src]  # [num_edges, attn_dim]
                    dst_causal = x_causal_proj[dst]  # [num_edges, attn_dim]
                    causal_attn = torch.cat([src_causal, dst_causal], dim=-1) @ self.causal_attn_vec

                    # 赋值关系类型的特定权重
                    rel_weight = self.relation_attn[i]

                    # 组合关联性和因果性注意力
                    if self.causal_strength:
                        # 计算因果强度系数
                        causal_factor = torch.sigmoid(self.causal_strength_weights[i])
                        # 采用因果强度系数，对关联性和因果性进行加权组合
                        alpha = (1 - causal_factor) * corr_attn + causal_factor * causal_attn + rel_weight
                        # 记录因果分数以便后续分析
                        causal_score = causal_factor * torch.sigmoid(causal_attn)
                    else:
                        # 若没有因果强度控制，则简单平均组合
                        alpha = 0.5 * (corr_attn + causal_attn) + rel_weight

                    # [新增] 应用稀疏性约束 - 促进稀疏的因果关系学习
                    if self.training and self.sparsity > 0:
                        # 随机丢弃部分边，鼓励模型专注于重要的因果关系
                        mask = torch.rand_like(alpha) > self.sparsity
                        alpha = alpha * mask

                    # 生成最终的边权重
                    edge_weight = torch.sigmoid(alpha).view(-1)

            # 消息传递：在原本RGCN的基础上增加了边权重
            if is_discrete:
                h = self.propagate(sub_edges, x=weight[i, x_l], edge_weight=edge_weight, size=size)
                out.add_(h)
            else:
                h = self.propagate(sub_edges, x=x_l, edge_weight=edge_weight, size=size)
                out.add_(h @ weight[i])

            # 保存因果分数
            if causal_score is not None:
                causal_scores.append((i, causal_score))

        # 11. 处理自连接权重与添加偏置
        if self.root is not None:
            if not torch.is_floating_point(x_r):
                out.add_(self.root[x_r])
            else:
                out.add_(x_r @ self.root)

        if self.bias is not None:
            out.add_(self.bias)

        return out

    def message(self, x_j, edge_weight=None):
        """
        定义节点间传递的信息形式

        在RGCN基础上增加了边权重的处理
        """
        # 如果有边权重，应用到消息上
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """
        组合消息生成和聚合步骤（与RGCN类似）
        """
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def do_intervention(self, x, edge_index, edge_type, target_nodes):
        """
        新增：执行因果干预操作（标准RGCN没有）

        实现do-calculus操作，通过移除与目标节点相关的边，分析干预效应

        参数:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            target_nodes: 要干预的目标节点列表

        返回:
            包含原始预测、干预后预测及其差异的字典
        """
        # 原始预测（无干预）
        original = self.forward(x, edge_index, edge_type)

        # 干预后预测
        intervened = self.forward(x, edge_index, edge_type, intervention_nodes=target_nodes)

        # 计算干预效应（因果影响）
        effect = original - intervened

        return {
            'original': original,  # 原始预测
            'intervened': intervened,  # 干预后预测
            'effect': effect  # 干预效应（衡量因果影响）
        }