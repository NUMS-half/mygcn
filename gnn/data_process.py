import dgl
import torch
from dgl import DGLGraph

# 定义节点类型
NODE_TYPES = {'User', 'Package', 'Behavior', 'Time', 'Rule'}