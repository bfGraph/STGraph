import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from seastar import Seastar
import torch.nn.functional as F
from seastar.backend.pytorch_backend import backend_cb
import snoop
import inspect

class SeastarGCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation = None,
                 dropout = None,
                 bias=True):
        super(SeastarGCNLayer, self).__init__()
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
        self.seastar = Seastar(backend_cb)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, edge_weight=None):
        if self.dropout:
            h = self.dropout(h)

        h = torch.mm(h, self.weight)

        @self.seastar.compile(nspace=[self, torch])
        def nb_compute(v):
            # The nb_edge.src returns a list with one element, this element is an object of NbNode type
            # hence the translation. Can be cleaned up later.
            h = sum([nb_edge.src[0].norm * nb_edge.src[0].h * nb_edge.weight for nb_edge in v.inedges])
            h = h * v.norm
            return h
        h = nb_compute(g=self.g, n_feats={'norm': self.norm, 'h' : h}, e_feats={'weight': edge_weight})

        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation is not None:
            h = self.activation(h)
        return h

class SeastarTGCNCell(torch.nn.Module):

    def __init__(
        self,
        g,
        in_channels: int,
        out_channels: int,
    ):
        super(SeastarTGCNCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g = g

        # Update GCN Layer
        self.conv_z = SeastarGCNLayer(self.g,self.in_channels,self.out_channels)
        
        # Update linear layer
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Reset GCN layer
        self.conv_r = SeastarGCNLayer(self.g, self.in_channels,self.out_channels)
        
        # Reset linear layer
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Candidate (Current Memory Content) GCN layer
        self.conv_h = SeastarGCNLayer(self.g, self.in_channels,self.out_channels)

        # Candidate linear layer
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_weight, H):
        h = self.conv_z(X, edge_weight=edge_weight)
        Z = torch.cat((h, H), axis=1) # axis values need to be checked
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_weight, H):
        R = torch.cat((self.conv_r(X, edge_weight=edge_weight), H), axis=1) # axis values need to be checked
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_weight, H, R):
        H_tilde = torch.cat((self.conv_h(X, edge_weight=edge_weight), H * R), axis=1) # axis values need to be checked
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.Tensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H

class SeastarTGCN(torch.nn.Module):
  def __init__(self, g, node_features):
    super(SeastarTGCN, self).__init__()
    self.g = g
    self.temporal = SeastarTGCNCell(self.g, node_features, 32)
    self.linear = torch.nn.Linear(32, 1)

  def forward(self, node_feat, edge_weight, hidden_state):
    h = self.temporal(node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    return y, h