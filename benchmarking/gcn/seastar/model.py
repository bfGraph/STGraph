import torch.nn as nn
from stgraph.nn.pytorch.static.gcn_conv import GCNConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(GCNConv(n_hidden, n_classes, None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h