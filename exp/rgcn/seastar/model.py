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
        self.num_rels = num_rels

        # List of weights for different relations
        # self.weight = [nn.Parameter(torch.Tensor(in_feats, out_feats)) for _ in range(num_rels)]
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feats, out_feats))
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
        # NOTE: this probably needs to be re-written
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, edge_norm, edge_types):
        if self.dropout:
            h = self.dropout(h)

        # Multiplying each relation with its corresponding weight matrix
        # NOTE: This can be optimized later
        # h = torch.stack([torch.mm(h,weight) for weight in self.weight])
        h = h.unsqueeze(0).expand(self.num_rels,-1,-1)
        h = torch.bmm(h,self.weight)
        h = h.permute(1, 0, 2)

        @self.cm.zoomIn(nspace=[self, torch])
        def nb_compute(v):
            h = sum([nb_edge.src[0].h for nb_edge in v.inedges])
            return h
        h = nb_compute(g=g, n_feats={'h' : h}, e_feats={'norm':edge_norm}, edge_types=edge_types)
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
                                 out_dim,
                                 num_rels)
        
        # self.layer2 = RGCNLayer(hidden_dim,
        #                          out_dim,
        #                          num_rels)
        
        self.emb = nn.Embedding(num_nodes, in_dim)
        
    def forward(self, g, feats, edge_norm, edge_types):
        feats = self.emb(feats)
        h = self.layer1(g, feats, edge_norm, edge_types)
        # h = F.relu(h)
        # h = self.layer2(g, h, edge_norm, edge_type)
        h = F.softmax(h, dim=1)
        return h