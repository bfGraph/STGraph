"""GCN using builtin functions that enables SPMV optimization.

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PyG_GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers):
        super(PyG_GCN, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, add_self_loops=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden, add_self_loops=False))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes, add_self_loops=False))

    def forward(self, features, edge_index):
        h = features
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        return h
