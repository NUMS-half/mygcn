import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
from torch_geometric.utils import dropout_edge
from torch.nn import LayerNorm

# 1. Base GCNConv - 2 layers
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.35, edge_dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.ln2 = LayerNorm(output_dim)
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):
        # 对边进行 dropout
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        return F.log_softmax(x, dim=1)

# 2. Base GCNConv - 3 layers
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 新增一层
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)


# # 3. GATConv - 2 layers
# from torch_geometric.nn import GATConv

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, heads=3, dropout=0.5):
#         super(GCN, self).__init__()
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
#         self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# # 4. SAGEConv - 2 layers
# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, edge_dropout=0.2):
#         super(GCN, self).__init__()
#         self.conv1 = SAGEConv(input_dim, hidden_dim)
#         self.conv2 = SAGEConv(hidden_dim, output_dim)
#         self.dropout = dropout
#         self.edge_dropout = edge_dropout
#
#     def forward(self, x, edge_index):
#         # 对边进行 dropout
#         edge_index, _ = dropout_adj(edge_index, p=self.edge_dropout, training=self.training)
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)