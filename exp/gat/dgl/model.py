
import argparse

import torch as th
import dgl
from torch import nn
from dgl.nn import GATv2Conv

class DGL_GAT_Model(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(DGL_GAT_Model, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.activation = activation
        self.gat_layers = nn.ModuleList()

        # input projection (no residual)
        self.gat_layers.append(GATv2Conv(in_dim,num_hidden,num_heads=heads[0],feat_drop=feat_drop,negative_slope=negative_slope,residual=True,share_weights=True,bias=False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATv2Conv(num_hidden*heads[l-1],num_hidden,num_heads=heads[l],feat_drop=feat_drop,negative_slope=negative_slope,residual=True,share_weights=True,bias=False))
        # output projection
        self.gat_layers.append(GATv2Conv(num_hidden*heads[-2],num_classes,num_heads=heads[-1],feat_drop=feat_drop,negative_slope=negative_slope,residual=True,share_weights=True,bias=False))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
