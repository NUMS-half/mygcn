import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric import is_compiling
from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing  # GNN消息传递基类
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, spmm, index_sort
from torch_geometric.typing import SparseTensor, pyg_lib, torch_sparse, WITH_SEGMM
from torch_geometric.backend import use_segment_matmul, use_segment_matmul_heuristic


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


class RGCNConv(MessagePassing):
    """
    关系图卷积网络层实现

    支持多种关系类型的图卷积操作，可选使用基分解减少参数量。
    针对不同输入类型(离散/连续特征)和边索引表示(Tensor/SparseTensor)提供了优化路径。

    Args:
        in_channels: 输入特征维度，可以是整数或包含两个元素的元组
        out_channels: 输出特征维度
        num_relations: 关系(边)类型的数量
        num_bases: 基函数的数量，用于参数分解，如果为None则不使用分解
        aggr: 聚合函数类型，默认为 'mean'
        root_weight: 是否使用自连接权重，默认为True
        bias: 是否使用偏置，默认为True
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
            **kwargs
    ):
        # 设置聚合方式，并确保没有覆盖用户指定的值
        kwargs.setdefault('aggr', aggr)
        # 调用父类初始化，设置聚合维度为节点维度(0)
        super().__init__(node_dim=0, **kwargs)

        # 保存初始化参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        # 处理输入通道可能为元组的情况(用于双模型输入)
        self.in_channels_l = in_channels[0] if isinstance(in_channels, tuple) else in_channels

        # 根据是否使用基分解创建权重参数
        if num_bases is not None and num_bases < num_relations:
            # 使用基分解以减少参数量: num_bases < num_relations
            self.weight = Parameter(torch.empty(num_bases, self.in_channels_l, out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))
        else:
            # 不使用分解或基数量等于关系数量
            self.weight = Parameter(torch.empty(num_relations, self.in_channels_l, out_channels))
            self.register_parameter('comp', None)  # 没有分解参数

        # 可选创建自连接权重和偏置
        self.root = Parameter(torch.empty(self.in_channels_l, out_channels)) if root_weight else None
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        # 初始化所有参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化/重置所有可学习参数为初始值"""
        super().reset_parameters()  # 重置消息传递参数
        glorot(self.weight)  # 权重初始化
        glorot(self.comp)  # 基分解系数初始化
        glorot(self.root)  # 自连接权重初始化
        zeros(self.bias)  # 偏置初始化为0

    def forward(self, x, edge_index, edge_type=None):
        """
        前向传播函数

        Args:
            x: 节点特征，可以是张量或包含两个元素的元组(用于源节点和目标节点不同特征)
            edge_index: 边索引，可以是形状为[2, num_edges]的张量或SparseTensor
            edge_type: 每条边的关系类型ID，形状为[num_edges]的整型张量

        Returns:
            形状为[num_nodes, out_channels]的节点表示更新

        优化点:
            1. 使用add_()原地操作减少内存分配
            2. 空关系集使用continue跳过，避免无效计算
            3. 根据特征类型选择最优处理路径
            4. 支持分段矩阵乘法加速(如果可用)
        """
        # 处理输入特征，支持元组输入(源和目标节点使用不同特征)
        x_l = x[0] if isinstance(x, tuple) else x  # 源节点特征
        x_r = x[1] if isinstance(x, tuple) else x_l  # 目标节点特征(默认与源相同)

        # 处理离散特征情况(特征为节点ID索引)
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        # 从稀疏张量中提取边类型(如果边索引是稀疏张量)
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None  # 确保边类型可用

        # 初始化输出张量，全零预分配提高性能
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        size = (x_l.size(0), x_r.size(0))  # 源和目标节点数量

        # 处理权重 - 如果使用基分解，合成完整权重
        weight = self.weight
        if self.num_bases is not None and self.comp is not None:
            # 基函数合成: weight = comp @ weight
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # 决定是否使用分段矩阵乘法优化
        use_segmm = use_segment_matmul  # 全局设置值
        if use_segmm is None and isinstance(edge_index, Tensor):
            # 计算num_relations种关系类型的边数量
            segment_count = scatter(torch.ones_like(edge_type), edge_type, dim_size=self.num_relations)
            # 根据关系数量、最大分段大小和特征维度决定是否使用分段优化
            use_segmm = use_segment_matmul_heuristic(
                num_segments=self.num_relations,
                max_segment_size=int(segment_count.max()),
                in_channels=weight.size(1),
                out_channels=weight.size(2),
            )

        # 分段矩阵乘法优化路径 - 一次性处理所有关系类型
        if (use_segmm and WITH_SEGMM and not is_compiling()
                and x_l.is_floating_point() and isinstance(edge_index, Tensor)):
            # 边类型需要排序以使用分段矩阵乘法
            if (edge_type[1:] < edge_type[:-1]).any():
                edge_type, perm = index_sort(edge_type, max_value=self.num_relations)
                edge_index = edge_index[:, perm]  # 相应地重排边索引

            # 创建指向每个关系类型边界的指针数组
            edge_type_ptr = index2ptr(edge_type, self.num_relations)
            # 一次调用propagate处理所有关系
            out = self.propagate(edge_index, x=x_l, edge_type_ptr=edge_type_ptr, size=size)
        else:
            # 常规处理路径 - 按关系类型逐个处理
            # 判断特征是离散的还是连续的
            is_discrete = not torch.is_floating_point(x_l)

            # 针对每种关系类型分别处理
            for i in range(self.num_relations):
                # 创建当前关系类型的边掩码
                mask = edge_type == i
                if not mask.any():  # 如果没有该类型的边，跳过
                    continue

                # 根据关系类型筛选边索引
                # 对于当前关系类型i, mask包含了所有edge_type == i的边的布尔掩码
                # masked_edge_index函数会根据这个掩码返回只包含当前关系类型的边子集
                # 返回的tmp保持与原始edge_index相同的格式(Tensor或SparseTensor)
                # 这样可以为每种关系类型分别执行消息传递，实现RGCN的关系特异性特征转换
                tmp = masked_edge_index(edge_index, mask)

                if is_discrete:
                    # 离散特征处理路径 - 直接在传播前应用权重
                    out.add_(self.propagate(tmp, x=weight[i, x_l], edge_type_ptr=None, size=size))
                else:
                    # 连续特征处理路径 - 先传播再应用权重
                    h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                    out.add_(h @ weight[i])  # 应用关系特定权重

        # 处理自连接(自环)权重
        if self.root is not None:
            if not torch.is_floating_point(x_r):
                out.add_(self.root[x_r])  # 离散特征的自连接处理
            else:
                out.add_(x_r @ self.root)  # 连续特征的自连接处理

        # 添加偏置项(如果有)
        if self.bias is not None:
            out.add_(self.bias)

        return out

    def message(self, x_j, edge_type_ptr=None):
        """
        定义消息函数，确定如何从源节点生成消息

        Args:
            x_j: 源节点特征
            edge_type_ptr: 分段矩阵乘法使用的边类型指针

        Returns:
            处理后的消息
        """
        # 如果启用了分段矩阵乘法优化且提供了边类型指针
        if WITH_SEGMM and not is_compiling() and edge_type_ptr is not None:
            # 使用优化版本一次性为所有边计算消息
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)
        # 默认情况下直接返回源节点特征
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """
        组合消息生成和聚合步骤，用于稀疏表示的优化

        Args:
            adj_t: 转置的邻接矩阵(SparseTensor)
            x: 节点特征

        Returns:
            聚合后的消息
        """
        # 确保邻接矩阵没有边权重(使用拓扑结构)
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        # 执行稀疏矩阵乘法，根据self.aggr指定的方式聚合
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        """返回层的字符串表示，用于打印模型结构"""
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
