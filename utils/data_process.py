import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils.helper import set_seed
from utils.logger import get_logger
from torch_geometric.data import Data
from utils.data_config import DIMENSION_MAPPINGS


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
        self.status_encoding = {}
        self.logger = get_logger("GraphGenerator")  # 配置日志

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

    def split_data(self, train_ratio=0.6, val_ratio=0.2):
        """
        划分数据集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            train_df, val_df, test_df
        """

        df = self.df
        total_len = len(df)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        self.logger.info(
            f"数据集已划分完成, Train：{len(train_df)}, Val：{len(val_df)}, Test：{len(test_df)}")

        return train_df, val_df, test_df

    @staticmethod
    def parse_combined_labels(lab_combined_str):
        """解析组合标签字符串为标签列表"""
        cleaned = lab_combined_str.strip('"')
        parts = cleaned.split('","')
        return [part.strip('"') for part in parts]

    def generate_triples(self, df):
        """从指定的 DataFrame 生成三元组数据"""
        # 初始化每个维度的三元组列表
        triples = {dimension: [] for dimension in DIMENSION_MAPPINGS}

        # 预处理：创建标签到维度的映射以提高性能
        label_dimension_map = {}
        for dimension, labels in DIMENSION_MAPPINGS.items():
            for label in labels:
                label_dimension_map[label] = dimension

        # 处理每行数据
        for _, row in df.iterrows():
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

        return triples

    def save_triples_to_dir(self, triples, output_dir):
        """将三元组保存到指定目录"""
        for dimension, dimension_triples in triples.items():
            if not dimension_triples:
                self.logger.warning(f"维度 '{dimension}' 没有三元组数据")
                continue

            triples_df = pd.DataFrame(
                dimension_triples,
                columns=["OrderID", "Label", "StatusCode"]
            )
            file_name = os.path.join(output_dir, f"user_{dimension}_triples.csv")

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

    @staticmethod
    def build_knowledge_graph(triples, df):
        set_seed()
        graph = nx.Graph()

        # 创建映射
        order_attributes = {}
        for _, row in df.iterrows():
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
        edges_to_add = set()

        for dimension, dimension_triples in triples.items():
            for order_id, label, status_code in dimension_triples:
                if label == 1:
                    nodes_to_add.add((order_id, 'order'))
                    nodes_to_add.add((status_code, 'status'))
                    edges_to_add.add((order_id, status_code))

        # 批量添加节点和边
        for node, node_type in nodes_to_add:
            if node_type == 'order' and node in order_attributes:
                graph.add_node(node, type=node_type, **order_attributes[node])
            else:
                graph.add_node(node, type=node_type)

        for u, v in edges_to_add:
            graph.add_edge(u, v, label=1)

        return graph

    # def save_graph_as_pt(self, graph, output_dir, dataset_type="train", filename="knowledge_graph.pt"):
    #     """
    #     将构建的图保存为 PyTorch Geometric 格式，并生成训练/验证/测试掩码
    #
    #     Args:
    #         graph: NetworkX图
    #         output_dir: 输出目录
    #         dataset_type: 数据集类型 ("train", "val", "test")
    #         filename: 输出文件名
    #     """
    #     edge_index = []
    #     edge_attr = []
    #     node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    #     behavior_labels = []  # 存储节点行为标签信息
    #     time_attrs = []  # 存储时间属性
    #     process_attrs = []  # 存储流程属性
    #
    #     # 收集节点属性
    #     for node in graph.nodes():
    #         node_attr = graph.nodes[node]
    #         # 收集行为标签
    #         if node_attr.get('type') == 'order' and 'behavior_label' in node_attr:
    #             behavior_labels.append(int(node_attr['behavior_label']))
    #         else:
    #             behavior_labels.append(-1)  # 对于非订单节点或没有行为标签的节点
    #
    #         # 收集时间和流程属性
    #         if node_attr.get('type') == 'order':
    #             time_attrs.append(node_attr.get('time', ''))
    #             process_attrs.append(node_attr.get('process', ''))
    #         else:
    #             time_attrs.append('')
    #             process_attrs.append('')
    #
    #     # 为节点分配ID并更新边的索引
    #     for u, v, data in graph.edges(data=True):
    #         edge_index.append([node_mapping[u], node_mapping[v]])
    #         edge_attr.append([data['label']])
    #
    #     if not edge_index:  # 检查是否有边
    #         self.logger.warning(f"图中没有边，跳过保存 {filename}")
    #         return
    #
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #
    #     # 节点特征 (1类用户 + 42类行为)
    #     node_features = []
    #     for node in graph.nodes():
    #         node_attr = graph.nodes[node]
    #         if node_attr.get('type') == 'order':
    #             # 用户节点 - 设置第一维为1，其余为0
    #             features = [0.0] * 43
    #             features[0] = 1.0  # 用户特征标识
    #             node_features.append(features)
    #         else:  # status节点
    #             # 状态节点 - 对应状态码设置为1
    #             features = [0.0] * 43
    #             try:
    #                 status_code = int(node)  # 节点ID实际是状态码
    #                 if 0 <= status_code < 42:  # 确保状态码在有效范围内
    #                     features[status_code + 1] = 1.0  # 从索引1开始表示42种行为
    #             except ValueError:
    #                 pass  # 非数字状态码使用全 0 特征
    #             node_features.append(features)
    #
    #     node_features = torch.tensor(node_features, dtype=torch.float)
    #
    #     # 创建行为标签张量
    #     behavior_labels = torch.tensor(behavior_labels, dtype=torch.long)
    #
    #     # 生成掩码
    #     num_nodes = len(graph.nodes)
    #     train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #
    #     # 根据数据集类型，为所有节点设置相应的掩码
    #     # 对于训练集，只有训练掩码为真
    #     # 对于验证集，只有验证掩码为真
    #     # 对于测试集，只有测试掩码为真
    #     if dataset_type == "train":
    #         train_mask.fill_(True)
    #     elif dataset_type == "val":
    #         val_mask.fill_(True)
    #     else:  # test
    #         test_mask.fill_(True)
    #
    #     # 创建PyTorch Geometric数据对象
    #     data = Data(
    #         x=node_features,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr,
    #         y=behavior_labels,
    #         train_mask=train_mask,
    #         val_mask=val_mask,
    #         test_mask=test_mask
    #     )
    #
    #     # 保存时间和流程属性（作为元数据）
    #     time_dict = {i: attr for i, attr in enumerate(time_attrs) if attr}
    #     process_dict = {i: attr for i, attr in enumerate(process_attrs) if attr}
    #     data.time_attr = time_dict
    #     data.process_attr = process_dict
    #
    #     # 保存为 .pt 文件
    #     file_path = os.path.join(output_dir, filename)
    #     torch.save(data, file_path)
    #     self.logger.info(f"图数据已保存为: {file_path}")

    def save_graph_as_pt(self, graph, output_dir, dataset_type="train", filename="knowledge_graph.pt"):
        """
        将构建的图保存为 PyTorch Geometric 格式，并生成训练/验证/测试掩码

        Args:
            graph: NetworkX图
            output_dir: 输出目录
            dataset_type: 数据集类型 ("train", "val", "test")
            filename: 输出文件名
        """
        # 1. 确定性节点排序和映射
        all_nodes = sorted(list(graph.nodes()), key=lambda x: str(x))
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

        # 2. 收集节点属性（按排序顺序）
        behavior_labels = []
        time_attrs = []
        process_attrs = []
        node_types = []

        for node in all_nodes:
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

        # 3. 构建节点特征矩阵（按排序顺序）
        node_features = []
        for i, node in enumerate(all_nodes):
            features = [0.0] * 43
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

        # 4. 确定性边处理 - 关键改进部分
        src_nodes = []
        dst_nodes = []
        edge_attrs = []

        # 收集所有边并标准化源-目标关系
        edges_data = []
        for u, v, data in graph.edges(data=True):
            src_idx = node_mapping[u]
            dst_idx = node_mapping[v]
            label = data.get('label', 1)
            edges_data.append((src_idx, dst_idx, label))

        # 确定性排序边
        edges_data.sort()

        # 分离源节点、目标节点和边属性
        for src, dst, label in edges_data:
            src_nodes.append(src)
            dst_nodes.append(dst)
            edge_attrs.append([float(label)])

        # 5. 创建确定性张量
        # 直接创建[2, num_edges]格式的edge_index，避免使用t()
        if src_nodes:
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            self.logger.warning(f"图中没有边，跳过保存 {filename}")
            return

        node_features = torch.tensor(node_features, dtype=torch.float)
        behavior_labels = torch.tensor(behavior_labels, dtype=torch.long)

        # 6. 掩码生成（按排序后的节点）
        num_nodes = len(all_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if dataset_type == "train":
            train_mask.fill_(True)
        elif dataset_type == "val":
            val_mask.fill_(True)
        else:  # test
            test_mask.fill_(True)

        # 7. 创建PyG数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=behavior_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        # 8. 保存元数据
        time_dict = {i: attr for i, attr in enumerate(time_attrs) if attr}
        process_dict = {i: attr for i, attr in enumerate(process_attrs) if attr}
        data.time_attr = time_dict
        data.process_attr = process_dict

        # 9. 保存为.pt文件
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        torch.save(data, file_path)
        self.logger.info(f"图数据已保存为: {file_path}")

        # 10. 输出验证信息用于调试（可选）
        self.logger.debug(f"节点数: {num_nodes}, 边数: {len(src_nodes)}")
        self.logger.debug(f"edge_index shape: {edge_index.shape}")

    @staticmethod
    def visualize_graph(graph, dataset_name, save=False):
        """可视化知识图谱"""
        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []

        for node, data in graph.nodes(data=True):
            if data['type'] == 'order':  # 订单ID节点
                node_colors.append('lightgreen')
                node_sizes.append(300)  # 订单ID节点的大小
            elif data['type'] == 'status':  # 状态编码节点
                node_colors.append('skyblue')
                node_sizes.append(150)  # 状态编码节点的大小

        # 绘制图
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph, seed=42, k=0.1)  # 使用spring布局，k决定节点之间的间距
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10,
                font_weight='bold', edge_color='gray')

        plt.title(f"Knowledge Graph - {dataset_name}", fontsize=14)
        plt.show()
        if save:
            output_file = os.path.join("../data/processed", f"knowledge_graph_{dataset_name}.png")
            plt.savefig(output_file)
            print(f"图谱已保存为: {output_file}")

    def process_dataset(self, df, output_dir, dataset_name):
        """处理单个数据集的所有步骤"""
        set_seed()
        # 生成三元组
        triples = self.generate_triples(df)
        self.logger.info(f"已生成 {dataset_name} 数据集的三元组")

        # 保存三元组(可选)
        # self.save_triples_to_dir(triples, output_dir)

        # 构建知识图谱
        graph = self.build_knowledge_graph(triples, df)
        self.logger.info(f"已构建 {dataset_name} 数据集的知识图谱")

        # 保存图
        self.save_graph_as_pt(graph, output_dir, dataset_name)

        # 可视化图谱(可选)
        # self.visualize_graph(graph, dataset_name)

        return triples

    def process(self):
        """执行完整的三元组生成流程"""
        try:
            set_seed()
            # 1. 加载数据，初始化编码
            self.load_data()  # 加载数据
            self.initialize_encoding()  # 初始化编码
            self.save_encoding_info()  # 保存编码信息

            # 2. 划分数据集
            train_df, val_df, test_df = self.split_data()

            # 3. 为每个数据集创建目录
            train_dir = os.path.join(self.output_dir, "train")
            val_dir = os.path.join(self.output_dir, "val")
            test_dir = os.path.join(self.output_dir, "test")

            for directory in [train_dir, val_dir, test_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    self.logger.info(f"创建数据集目录: {directory}")

            # 4. 处理训练集
            self.logger.info(f"{'-' * 20}处理训练集数据{'-' * 20}")
            self.process_dataset(train_df, train_dir, "train")

            # 5. 处理验证集
            self.logger.info(f"{'-' * 20}处理验证集数据{'-' * 20}")
            self.process_dataset(val_df, val_dir, "val")

            # 6. 处理测试集
            self.logger.info(f"{'-' * 20}处理测试集数据{'-' * 20}")
            self.process_dataset(test_df, test_dir, "test")

            return True
        except Exception as e:
            self.logger.error(f"处理过程中出错: {e}")
            return False


# 执行三元组生成并构建知识图谱
def main():
    generator = GraphGenerator()
    generator.process()


if __name__ == "__main__":
    main()
