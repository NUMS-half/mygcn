import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from utils.helper import set_seed, get_logger
from utils.data_config import DIMENSION_MAPPINGS, DIMENSION_TO_RELATION


class GraphGenerator:
    """
    三元组数据与可视化图谱生成器类
    用于将用户行为数据转换为三元组格式(OrderID, Label, StatusCode)
    """

    def __init__(self, input_file="../data/raw/labeled_user_behavior_data.csv",
                 output_dir="../data/processed"):
        """
        初始化三元组生成器

        Args:
            input_file: 输入CSV文件路径
            output_dir: 输出文件目录
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.triples = None
        self.status_encoding = {}
        self.logger = get_logger("GraphGenerator")

    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.input_file, encoding='utf-8-sig')
            self.logger.info(f"成功加载数据: {self.input_file}, 共{len(self.df)}条记录")
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    def initialize_encoding(self):
        """为每个维度初始化独立的编码空间"""
        start_id = 0
        for dimension, labels in DIMENSION_MAPPINGS.items():
            # 为当前维度分配编码
            for i, label in enumerate(labels):
                self.status_encoding[f"{dimension}_{label}"] = start_id + i

            # 下一个维度的起始编码
            start_id += len(labels)

        self.logger.info(f"状态编码初始化完成，共{len(self.status_encoding)}个编码")

    def get_status_code(self, dimension, label):
        """获取特定维度和标签的编码值"""
        return self.status_encoding.get(f"{dimension}_{label}")

    @staticmethod
    def parse_combined_labels(lab_combined_str):
        """解析组合标签字符串为标签列表"""
        cleaned = lab_combined_str.strip('"')
        parts = cleaned.split('","')
        return [part.strip('"') for part in parts]

    def generate_triples(self):
        """从 self.df 生成三元组数据"""
        # 初始化每个维度的三元组列表
        triples = {dimension: [] for dimension in DIMENSION_MAPPINGS}

        # 预处理：创建标签到维度的映射以提高性能
        label_dimension_map = {}
        for dimension, labels in DIMENSION_MAPPINGS.items():
            for label in labels:
                label_dimension_map[label] = dimension

        # 处理每行数据
        for _, row in self.df.iterrows():
            order_id = row['DYPT.ORDER_ID']
            labels = self.parse_combined_labels(row['DYPT.LAB_NAM_COMBINED'])

            # 收集每个维度的编码
            dimension_codes = {}

            # 通过标签内容匹配维度（使用优化的映射）
            for label in labels:
                if label in label_dimension_map:
                    dimension = label_dimension_map[label]
                    dimension_codes[dimension] = self.get_status_code(dimension, label)

            # 为每个维度生成三元组
            for dimension, labels in DIMENSION_MAPPINGS.items():
                if dimension in dimension_codes and dimension_codes[dimension] is not None:
                    # 添加正样本
                    triples[dimension].append((order_id, 1, dimension_codes[dimension]))

                    # 添加负样本
                    for label in labels:
                        other_code = self.get_status_code(dimension, label)
                        if other_code != dimension_codes[dimension]:
                            triples[dimension].append((order_id, 0, other_code))

        self.triples = triples
        self.logger.info(f"已生成数据集的三元组")
        return triples

    def save_triples_to_dir(self):
        """将三元组保存到指定目录"""
        for dimension, dimension_triples in self.triples.items():
            if not dimension_triples:
                self.logger.warning(f"维度 '{dimension}' 没有三元组数据")
                continue

            triples_df = pd.DataFrame(
                dimension_triples,
                columns=["OrderID", "Label", "StatusCode"]
            )
            file_name = os.path.join(self.output_dir, f"user_{dimension}_triples.csv")

            try:
                triples_df.to_csv(file_name, index=False, encoding='utf-8-sig')
                self.logger.info(f"保存三元组数据: {file_name}, 共{len(dimension_triples)}条记录")
            except Exception as e:
                self.logger.error(f"保存三元组失败 '{dimension}': {e}")

    def save_encoding_info(self):
        """将标签编码映射信息保存为CSV文件"""
        # 准备映射数据
        mapping_data = []
        for dimension, labels in DIMENSION_MAPPINGS.items():
            for label in labels:
                code = self.get_status_code(dimension, label)
                mapping_data.append([dimension, label, code])

        # 创建DataFrame并保存：维度状态与编码的映射关系
        mapping_df = pd.DataFrame(mapping_data, columns=["Dimension", "Label", "Code"])
        file_name = os.path.join(self.output_dir, "encoding_mapping.csv")

        try:
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.logger.info(f"已创建输出目录: {self.output_dir}")

            mapping_df.to_csv(file_name, index=False, encoding='utf-8-sig')
            self.logger.info(f"编码映射信息已保存至: {file_name}")
        except Exception as e:
            self.logger.error(f"保存编码映射信息失败: {e}")


    def build_knowledge_graph(self):
        graph = nx.Graph()

        # 创建映射
        order_attributes = {}
        for _, row in self.df.iterrows():
            order_id = row['DYPT.ORDER_ID']
            behavior_label = row.get('BEHAVIOR_LABEL', None)
            accept_dt = row.get('DYPT.ACCEPT_DT', "")
            ms_process = row.get('DYPT.MS', "")

            order_attributes[order_id] = {
                'behavior_label': int(behavior_label) if pd.notna(behavior_label) else -1,
                'time': accept_dt,
                'process': ms_process
            }

        # 预处理：收集所有需要添加的节点和边避免重复
        nodes_to_add = set()
        edges_to_add = []  # 使用列表存储边及其属性

        for dimension, dimension_triples in self.triples.items():
            for order_id, label, status_code in dimension_triples:
                if label == 1:
                    nodes_to_add.add((order_id, 'order'))
                    nodes_to_add.add((status_code, 'status'))

                    # 获取该dimension对应的关系类型
                    relation_type = DIMENSION_TO_RELATION.get(dimension, -1)
                    edges_to_add.append((order_id, status_code, relation_type))

        # 批量添加节点和边
        for node, node_type in nodes_to_add:
            if node_type == 'order' and node in order_attributes:
                graph.add_node(node, type=node_type, **order_attributes[node])
            else:
                graph.add_node(node, type=node_type)

        for u, v, relation_type in edges_to_add:
            graph.add_edge(u, v, label=1, relation_type=relation_type)

        for status_code in range(42):
            if status_code in graph.nodes():
                node_edges = list(graph.edges(status_code))
                self.logger.info(f"特征节点 {status_code} 有 {len(node_edges)} 条边")
            else:
                self.logger.info(f"特征节点 {status_code} 不存在于图中")

        self.logger.info(f"已基于数据集构建知识图谱")

        return graph

    def save_graph_as_pt(self, graph, train_ratio, val_ratio, filename="knowledge_graph.pt"):
        """将构建的图保存为 PyTorch Geometric 格式，并按时间顺序生成训练/验证/测试掩码"""
        # 1. 修改节点映射逻辑，确保特征节点保持原ID
        node_mapping = {}
        # 先映射特征节点(0-41)，确保这些节点保持原有ID
        for i in range(42):
            if i in graph.nodes():
                node_mapping[i] = i

        # 再处理其他节点
        next_id = 42
        for node in graph.nodes():
            if node not in node_mapping:
                node_mapping[node] = next_id
                next_id += 1

        # 2. 创建按映射排序的节点列表（用于后续处理）
        max_idx = max(node_mapping.values()) if node_mapping else 0
        all_nodes = [None] * (max_idx + 1)

        for node, idx in node_mapping.items():
            all_nodes[idx] = node

        # 3. 收集节点属性
        behavior_labels = []
        time_attrs = []
        process_attrs = []
        node_types = []

        for node_idx in range(len(all_nodes)):
            node = all_nodes[node_idx]
            if node is None:
                # 对于不存在的节点，添加默认值
                behavior_labels.append(-1)
                time_attrs.append('')
                process_attrs.append('')
                node_types.append('')
                continue

            node_attr = graph.nodes[node]
            node_type = node_attr.get('type', '')
            node_types.append(node_type)

            # 收集行为标签
            if node_type == 'order' and 'behavior_label' in node_attr:
                behavior_labels.append(int(node_attr['behavior_label']))
            else:
                behavior_labels.append(-1)

            # 收集时间和流程属性
            if node_type == 'order':
                time_attrs.append(node_attr.get('time', ''))
                process_attrs.append(node_attr.get('process', ''))
            else:
                time_attrs.append('')
                process_attrs.append('')

        # 4. 构建节点特征矩阵
        node_features = []
        for i in range(len(all_nodes)):
            node = all_nodes[i]
            features = [0.0] * 43

            if node is not None:
                if node_types[i] == 'order':
                    features[0] = 1.0  # 用户特征标识
                else:
                    try:
                        status_code = int(node)
                        if 0 <= status_code < 42:
                            features[status_code + 1] = 1.0
                    except (ValueError, TypeError):
                        pass
            node_features.append(features)

        # 5. 边处理 - 修复关键区域
        edge_list = []
        edge_attrs = []
        edge_types = []

        # 遍历原始图中的所有边
        for u, v, data in graph.edges(data=True):
            src_idx = node_mapping[u]
            dst_idx = node_mapping[v]
            label = data.get('label', 1)
            relation_type = data.get('relation_type', -1)

            # PyG 对于无向图采用双向存储
            edge_list.append((src_idx, dst_idx))
            edge_attrs.append([float(label)])
            edge_types.append(relation_type)

            edge_list.append((dst_idx, src_idx))
            edge_attrs.append([float(label)])
            edge_types.append(relation_type)

        # 6. 创建确定性张量
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            self.logger.warning(f"图中没有边，跳过保存 {filename}")
            return

        node_features = torch.tensor(node_features, dtype=torch.float)
        behavior_labels = torch.tensor(behavior_labels, dtype=torch.long)

        # 7. 按时间顺序生成掩码
        num_nodes = len(all_nodes)
        order_indices = []
        non_order_indices = []

        # 收集订单节点和非订单节点
        for i, node_type in enumerate(node_types):
            if node_type == 'order' and time_attrs[i]:
                order_indices.append((i, time_attrs[i]))
            else:
                non_order_indices.append(i)

        # 按时间排序订单节点
        order_indices.sort(key=lambda x: x[1])
        sorted_order_indices = [idx for idx, _ in order_indices]

        # 按时间顺序划分订单节点
        order_count = len(sorted_order_indices)
        train_order_size = int(order_count * train_ratio)
        val_order_size = int(order_count * val_ratio)

        train_order_indices = sorted_order_indices[:train_order_size]
        val_order_indices = sorted_order_indices[train_order_size:train_order_size + val_order_size]
        test_order_indices = sorted_order_indices[train_order_size + val_order_size:]

        # 8. 创建掩码
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # 设置订单节点掩码
        train_mask[train_order_indices] = True
        val_mask[val_order_indices] = True
        test_mask[test_order_indices] = True

        # 将所有非订单节点分配给训练集
        train_mask[non_order_indices] = True

        # 9. 创建PyG数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            y=behavior_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        # 10. 保存元数据
        time_dict = {i: attr for i, attr in enumerate(time_attrs) if attr}
        process_dict = {i: attr for i, attr in enumerate(process_attrs) if attr}
        data.time_attr = time_dict
        data.process_attr = process_dict

        # 11. 保存为.pt文件
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, filename)
        torch.save(data, file_path)

        # 12. 输出划分统计信息
        self.logger.info(f"图谱数据已保存为: {file_path}")
        self.logger.info(f"数据集划分: 订单节点 - 训练:{len(train_order_indices)}, 验证:{len(val_order_indices)}, 测试:{len(test_order_indices)}")
        self.logger.info(f"非订单节点(全部分配到训练集): {len(non_order_indices)}")

    def visualize_graph(self, graph, save=False, output_path="../data/processed/knowledge_graph.png"):
        """
        高级知识图谱可视化函数，支持多种节点类型和关系类型的区分显示

        Args:
            graph: NetworkX图对象
            save: 是否保存图像
            output_path: 图像保存路径
        """
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        plt.figure(figsize=(16, 12), facecolor='white')

        # 使用Fruchterman-Reingold布局，对小型图更清晰
        pos = nx.spring_layout(graph, k=1.2, iterations=200, seed=42)
        pos = {node: coord * 2.0 for node, coord in pos.items()}  # 扩大1.5倍

        # 获取所有节点类型和关系类型
        node_types = {data['type'] for _, data in graph.nodes(data=True)}
        relation_types = {data.get('relation_type', -1) for _, _, data in graph.edges(data=True)}

        # 为节点类型设置颜色映射
        node_type_colors = {
            'order': '#FF9999',  # 浅红色
            'status': '#66B2FF'  # 浅蓝色
        }

        # 为关系类型设置颜色和样式映射
        relation_colors = {
            0: '#E6194B',  # 红色
            1: '#3CB44B',  # 绿色
            2: '#FFA500',  # 橙色
            3: '#4363D8',  # 蓝色
            4: '#F58231',  # 浅橙色
            5: '#911EB4',  # 紫色
            6: '#46F0F0',  # 浅青色
            7: '#F032E6',  # 浅紫色
            8: '#BCF60C',  # 浅绿色
        }

        relation_styles = {
            0: 'solid',
            1: 'dashed',
            2: 'dotted',
            3: 'dashdot',
            4: (0, (3, 1, 1, 1)),  # 自定义虚线模式
            5: (0, (5, 1)),  # 自定义虚线模式
            6: (0, (1, 1)),  # 自定义虚线模式
            7: (0, (2, 1)),  # 自定义虚线模式
            8: (0, (1, 2)),  # 自定义虚线模式
            -1: 'solid'
        }

        relation_widths = {k: 1.5 + 0.5 * k if k >= 0 else 1.0 for k in relation_types}

        # 为节点设置大小和标签
        node_sizes = []
        node_colors = []
        node_shapes = []
        node_labels = {}

        for node, data in graph.nodes(data=True):
            node_type = data['type']
            if node_type == 'order':
                size = 800  # 订单节点更大
                shape = 'o'  # 圆形
                # 创建简洁标签
                node_labels[node] = f"{str(node)[:8]}..."
            else:  # status
                size = 500
                shape = 's'  # 方形
                node_labels[node] = str(node)

            node_sizes.append(size)
            node_colors.append(node_type_colors[node_type])
            node_shapes.append(shape)

        # 绘制节点
        # 订单节点（圆形）
        order_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'order']
        order_sizes = [node_sizes[i] for i, (n, _) in enumerate(graph.nodes(data=True)) if n in order_nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=order_nodes,
                               node_color=node_type_colors['order'],
                               node_size=order_sizes, alpha=0.9,
                               node_shape='o', edgecolors='black', linewidths=1)

        # 状态节点（方形）
        status_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'status']
        status_sizes = [node_sizes[i] for i, (n, _) in enumerate(graph.nodes(data=True)) if n in status_nodes]
        nx.draw_networkx_nodes(graph, pos, nodelist=status_nodes,
                               node_color=node_type_colors['status'],
                               node_size=status_sizes, alpha=0.9,
                               node_shape='s', edgecolors='black', linewidths=1)

        # 分别绘制不同类型的边
        for relation_type in relation_types:
            edges = [(u, v) for u, v, d in graph.edges(data=True)
                     if d.get('relation_type', -1) == relation_type]
            if edges:
                nx.draw_networkx_edges(graph, pos, edgelist=edges,
                                       width=relation_widths[relation_type],
                                       edge_color=relation_colors[relation_type],
                                       style=relation_styles[relation_type],
                                       alpha=0.8,
                                       arrows=True,
                                       connectionstyle='arc3,rad=0.1')  # 弯曲的边，避免重叠

        # 绘制节点标签
        nx.draw_networkx_labels(graph, pos, labels=node_labels,
                                font_size=10, font_weight='bold',
                                font_family='sans-serif')

        # 添加图例
        # 节点类型图例
        node_handles = []
        node_labels = []
        for node_type, color in node_type_colors.items():
            if node_type == 'order':
                node_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                               markersize=12, markeredgecolor='black', markeredgewidth=1))
                node_labels.append('订单节点')
            else:
                node_handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                                               markersize=16, markeredgecolor='black', markeredgewidth=1))
                node_labels.append('状态/特征节点')

        # 关系类型图例
        relation_handles = []
        relation_labels = []
        relation_name_map = {
            0: "用户特征关系",
            1: "用户套餐关系",
            2: "用户行为关系",

            # # 用户特征关系
            # 0: "用户价值维度关系",
            # 1: "用户信用维度关系",
            # 2: "用户反馈维度关系",
            #
            # # 用户套餐关系
            # 3: "套餐月租费关系",
            # 4: "ARPU值关系",
            #
            # # 用户行为关系
            # 5: "流量使用关系",
            # 6: "语音超套费用关系",
            # 7: "流量超套费用关系",
            # 8: "MOU关系",

            -1: "未知关系类型"  # 添加默认关系类型
        }

        # 安全处理所有关系类型
        for rel_type in sorted(relation_types):
            if rel_type in relation_colors and rel_type in relation_styles:
                relation_handles.append(plt.Line2D([0], [0],
                                                   color=relation_colors.get(rel_type, 'gray'),
                                                   linestyle=relation_styles.get(rel_type, 'solid'),
                                                   lw=2))
                relation_labels.append(relation_name_map.get(rel_type, f"关系类型 {rel_type}"))

        # 合并图例
        all_handles = node_handles + relation_handles
        all_labels = node_labels + relation_labels
        plt.legend(all_handles, all_labels, loc='upper right', fontsize=14,
                   fancybox=True, framealpha=0.9, edgecolor='gray')

        # 设置标题和背景
        plt.title("知识图谱可视化", fontsize=24, fontweight='bold', pad=20)

        # 移除坐标轴
        plt.axis('off')

        # 添加边框
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        # 添加水印
        plt.figtext(0.5, 0.01, "知识图谱关系分析", ha="center", fontsize=10, color="gray")

        # 调整布局
        plt.tight_layout()

        # 保存和显示
        if save:
            plt.savefig(output_path, dpi=400, bbox_inches='tight', transparent=True)
            self.logger.info(f"图谱已保存为: {output_path}")

        plt.show()

        # 如果节点非常多，可以输出节点和关系统计信息
        self.logger.info(f"图谱统计: {graph.number_of_nodes()}个节点, {graph.number_of_edges()}条边")
        self.logger.info(f"节点类型: {', '.join(node_types)}")
        self.logger.info(f"关系类型: {', '.join(str(t) for t in sorted(relation_types))}")

    def process_dataset(self, train_ratio, val_ratio):
        """数据集处理步骤"""
        self.logger.info(f"{'=' * 20} 处理并划分集数据 {'=' * 20}")

        self.generate_triples()  # 生成三元组
        # self.save_triples_to_dir()  # 保存三元组(可选)
        graph = self.build_knowledge_graph()  # 构建知识图谱
        self.save_graph_as_pt(graph, train_ratio, val_ratio)  # 保存图谱(划分数据集)
        # self.visualize_graph(graph)  # 可视化图谱(可选)

    def process(self):
        """数据预处理全流程"""
        try:
            self.logger.info(f"{'=' * 20} 开始处理数据集 {'=' * 20}")

            self.load_data()  # 加载数据
            self.initialize_encoding()  # 初始化编码
            self.save_encoding_info()  # 保存编码信息
            self.process_dataset(train_ratio=0.7, val_ratio=0.15)  # 处理数据集

            self.logger.info(f"{'=' * 20} 数据集处理完成 {'=' * 20}")
            return True
        except Exception as e:
            self.logger.error(f"处理过程中出错: {e}")
            return False


# 执行三元组生成并构建知识图谱
def main():
    generator = GraphGenerator()
    generator.process()


if __name__ == "__main__":
    set_seed(45)
    main()
