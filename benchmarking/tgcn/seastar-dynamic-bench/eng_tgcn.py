import math
import torch
import torch.nn as nn
from seastar.compiler import Seastar
import torch.nn.functional as F
from seastar.compiler.backend.pytorch_backend import backend_cb
import snoop
from rich import inspect

from rich.traceback import install
install(show_locals=True)

class SeastarGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation = None,
                 dropout = None,
                 bias=True):
        super(SeastarGCNLayer, self).__init__()
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
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            nn.init.zeros_(self.bias)

    def forward(self, g, h, edge_weight=None):
        if self.dropout:
            h = self.dropout(h)

        h = torch.mm(h, self.weight)

        @self.seastar.compile(nspace=[self, torch])
        def nb_compute(v):
            h = sum([nb_edge.src.norm * nb_edge.src.h * nb_edge.edge_weight for nb_edge in v.inedges])
            h = h * v.norm
            return h
        
        # NOTE: THE VARIABLES THAT ARE PASSED HERE CANT HAVE THE SAME NAME AS ANY OF THE SELF VARIABLES DEFINED IN THIS CLASS
        # BECAUSE THEN THE VARIABLES IN INPUT_CACHE WOULD GET OVERWRITTEN
        h = nb_compute(g=g, n_feats={'norm': g.ndata['norm'], 'h' : h}, e_feats={'edge_weight': edge_weight})

        # @self.seastar.compile(nspace=[self, torch])
        # def nb_compute(v):
        #     # The nb_edge.src returns a list with one element, this element is an object of NbNode type
        #     # hence the translation. Can be cleaned up later.
        #     h = sum([nb.h*nb.norm for nb in v.innbs])
        #     h = h * v.norm
        #     return h
        # h = nb_compute(g=g, n_feats={'norm': g.ndata['norm'], 'h' : h})

        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation is not None:
            h = self.activation(h)
        return h

class SeastarTGCNCell(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(SeastarTGCNCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Update GCN Layer
        self.conv_z = SeastarGCNLayer(self.in_channels,self.out_channels)
        
        # Update linear layer
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Reset GCN layer
        self.conv_r = SeastarGCNLayer(self.in_channels,self.out_channels)
        
        # Reset linear layer
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Candidate (Current Memory Content) GCN layer
        self.conv_h = SeastarGCNLayer(self.in_channels,self.out_channels)

        # Candidate linear layer
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, g, X, edge_weight, H):
        # print(X.shape)
        h = self.conv_z(g, X, edge_weight=edge_weight)
        # print(h.shape)
        # print(H.shape)
        # print("\n")
        # quit()
        Z = torch.cat((h, H), axis=1) # axis values need to be checked
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, g, X, edge_weight, H):
        R = torch.cat((self.conv_r(g, X, edge_weight=edge_weight), H), axis=1) # axis values need to be checked
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, g, X, edge_weight, H, R):
        H_tilde = torch.cat((self.conv_h(g, X, edge_weight=edge_weight), H * R), axis=1) # axis values need to be checked
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        g,
        X: torch.Tensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(g, X, edge_weight, H)
        R = self._calculate_reset_gate(g, X, edge_weight, H)
        H_tilde = self._calculate_candidate_state(g, X, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
        

class Eng_SeastarTGCN(torch.nn.Module):
  def __init__(self, node_features):
    super(Eng_SeastarTGCN, self).__init__()
    self.temporal = SeastarTGCNCell(node_features, 32)
    self.linear = torch.nn.Linear(32, 1)

  def forward(self, g, node_feat, edge_weight, hidden_state):
    h = self.temporal(g, node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    
    return y, h