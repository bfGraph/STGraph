"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch
import time
import torch.nn as nn
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        ta = time.perf_counter()
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = self.propagate(self.g, h)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print('t1 - t0', t1-t0, 't0-ta',t0-ta, 'propagation ration', (t1-t0)/(t1-ta))
        return h
