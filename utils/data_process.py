import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils.logger import get_logger
from torch_geometric.data import Data
from utils.data_config import DIMENSION_MAPPINGS


class GraphGenerator:
    """
    三元组数据与可视化图谱生成器类
    用于将用户行为数据转换为适合机器学习的三元组格式(OrderID, Label, StatusCode)
    """

    def __init__(self, input_file="../data/raw/generated_user_behavior_data.csv",
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
        self.triples = {}
        self.graph = nx.Graph()  # 初始化图谱对象

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

    def parse_combined_labels(self, lab_combined_str):
        """解析组合标签字符串为标签列表"""
        cleaned = lab_combined_str.strip('"')
        parts = cleaned.split('","')
        return [part.strip('"') for part in parts]

    def generate_triples(self):
        """生成所有维度的三元组数据"""
        # 初始化每个维度的三元组列表
        self.triples = {dimension: [] for dimension in DIMENSION_MAPPINGS}

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
                    self.triples[dimension].append((order_id, 1, dimension_codes[dimension]))

                    # 添加负样本
                    for label in labels:
                        other_code = self.get_status_code(dimension, label)
                        if other_code != dimension_codes[dimension]:
                            self.triples[dimension].append((order_id, 0, other_code))

        self.logger.info("三元组生成完成")
        return self.triples

    def save_triples(self):
        """将三元组保存为CSV文件"""
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"创建输出目录: {self.output_dir}")

        # 为每个维度保存三元组
        for dimension, dimension_triples in self.triples.items():
            if not dimension_triples:  # 检查是否为空
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
        """根据三元组数据构建知识图谱"""

        # 通过三元组添加节点和边
        for dimension, dimension_triples in self.triples.items():
            for order_id, label, status_code in dimension_triples:
                # 仅添加 label 为 1 的边
                if label == 1:
                    # 添加订单ID和状态编码为节点
                    self.graph.add_node(order_id, type='order')  # 订单ID节点
                    self.graph.add_node(status_code, type='status')  # 状态编码节点

                    # 添加边（订单ID和状态编码之间的连接）
                    self.graph.add_edge(order_id, status_code, label=label)

        self.logger.info("知识图谱构建完成")

    def save_graph_as_pt(self, filename="knowledge_graph.pt"):
        """将构建的图保存为 PyTorch Geometric 格式"""
        edge_index = []
        edge_attr = []
        node_mapping = {node: idx for idx, node in enumerate(self.graph.nodes)}

        # 为节点分配ID并更新边的索引
        for u, v, data in self.graph.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])  # 将节点名称转为整数索引
            edge_attr.append([data['label']])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 节点特征
        num_nodes = len(self.graph.nodes)
        node_features = torch.eye(num_nodes, dtype=torch.float)  # 使用单位矩阵表示节点特征

        # 创建PyTorch Geometric数据对象
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        # 保存为 .pth 文件
        filename = os.path.join(self.output_dir, filename)
        torch.save(data, filename)
        self.logger.info(f"图数据已保存为: {filename}")

    def visualize_graph(self):
        """可视化知识图谱"""
        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []

        for node, data in self.graph.nodes(data=True):
            if data['type'] == 'order':  # 订单ID节点
                node_colors.append('lightgreen')
                node_sizes.append(300)  # 订单ID节点的大小
            elif data['type'] == 'status':  # 状态编码节点
                node_colors.append('skyblue')
                node_sizes.append(150)  # 状态编码节点的大小

        # 绘制图
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, seed=42, k=0.1)  # 使用spring布局，k决定节点之间的间距
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10,
                font_weight='bold', edge_color='gray')

        plt.title("Knowledge Graph of User Feedback Data", fontsize=14)
        plt.show()

    def process(self):
        """执行完整的三元组生成流程"""
        try:
            self.load_data()  # 加载数据
            self.initialize_encoding()  # 初始化编码
            self.save_encoding_info()  # 保存编码信息
            self.generate_triples()  # 生成三元组数据
            self.save_triples()  # 保存三元组文件
            self.build_knowledge_graph()  # 构建知识图谱
            self.save_graph_as_pt()  # 保存图为 .pt 文件
            # self.visualize_graph()  # 可视化知识图谱
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
