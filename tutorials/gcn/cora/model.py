from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from stgraph.nn.pytorch.static.gcn_conv import GCNConv
from stgraph.graph import StaticGraph


class GCN(nn.Module):
    def __init__(
        self: GCN,
        graph: StaticGraph,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_hidden_layers: int,
    ) -> None:
        super(GCN, self).__init__()

        self._graph = graph
        self._layers = nn.ModuleList()

        # input layer
        self._layers.append(GCNConv(in_feats, n_hidden, F.relu, bias=True))

        # hidden layers
        for i in range(n_hidden_layers):
            self._layers.append(GCNConv(n_hidden, n_hidden, F.relu, bias=True))

        # output layer
        self._layers.append(GCNConv(n_hidden, n_classes, None, bias=True))

    def forward(self: GCN, features):
        h = features
        for layer in self._layers:
            h = layer.forward(self._graph, h)
        return h
