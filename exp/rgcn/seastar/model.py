from torch import nn
import torch.nn.functional as F
import torch
from seastar import CtxManager
from seastar.backend.pytorch_backend import run_egl
import math

class RGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_rels,
                 activation=None,
                 dropout=None,
                 bias=True):
        super(RGCNLayer, self).__init__()
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
        self.cm = CtxManager(run_egl)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)

        h = torch.mm(h, self.weight)
        @self.cm.zoomIn(nspace=[self, torch])
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

class RGCNModel(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNModel, self).__init__()

        self.layer1 = RGCNLayer(in_dim,
                                 hidden_dim,
                                 num_rels,
                                 self_loop=False)
        
        self.layer2 = RGCNLayer(hidden_dim,
                                 out_dim,
                                 num_rels,
                                 self_loop=False)
        
        self.emb = nn.Embedding(num_nodes, in_dim)
        
    def forward(self, g, feats, edge_type, edge_norm):
        feats = self.emb(feats)
        h = self.layer1(g, feats, edge_type, edge_norm)
        h = F.relu(h)
        h = self.layer2(g, h, edge_type, edge_norm)
        h = F.softmax(h, dim=1)
        return h