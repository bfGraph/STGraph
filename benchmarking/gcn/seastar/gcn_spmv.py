"""GCN using builtin functions that enables SPMV optimization.

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import math
import torch
import torch.nn as nn
from seastar.compiler import Seastar
from seastar.compiler.backend.pytorch.torch_callback import SeastarBackendTorch

class EglGCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(EglGCNLayer, self).__init__()
        self.g = g
        self.norm = self.g.ndata['norm']
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
        self.seastar = Seastar(SeastarBackendTorch())

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            nn.init.zeros_(self.bias)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)

        h = torch.mm(h, self.weight)
        
        @self.seastar.compile(gnn_module=self)
        def nb_compute(v):
            h = sum([nb.h*nb.norm for nb in v.innbs])
            h = h * v.norm
            return h
        h = nb_compute(g=self.g, n_feats={'norm': self.norm, 'h' : h})
        
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class EglGCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(EglGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(EglGCNLayer(g, in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(EglGCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(EglGCNLayer(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h