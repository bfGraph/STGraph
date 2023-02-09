import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from seastar import CtxManager
import torch.nn.functional as F
from seastar.backend.pytorch_backend import run_egl
import snoop
import inspect

class SeastarGCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
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
        self.cm = CtxManager(run_egl)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    @snoop
    def forward(self, g, h, edge_weight=None):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)

        print("Shape")
        print(g.ndata['norm'].shape)

        @self.cm.zoomIn(nspace=[self, torch])
        @snoop
        def nb_compute(v):
            # The nb_edge.src returns a list with one element, this element is an object of NbNode type
            # hence the translation. Can be cleaned up later.
            h = sum([nb_edge.src[0].norm * nb_edge.src[0].h * nb_edge.weight for nb_edge in v.inedges])
            h = h * v.norm
            return h
        h = nb_compute(g=g, n_feats={'norm': g.ndata['norm'], 'h' : h}, e_feats={'weight': edge_weight})
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

# class SeastarTGCNCell(torch.nn.Module):
#     r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
#     For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
#     Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         add_self_loops (bool): Adding self-loops for smoothing. Default is True.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#     ):
#         super(SeastarTGCNCell, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Update GCN Layer
#         self.conv_z = nn.GraphConv(self.in_channels,self.out_channels)
        
#         # Update linear layer
#         self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#         # Reset GCN layer
#         self.conv_r = nn.GraphConv(self.in_channels,self.out_channels)
        
#         # Reset linear layer
#         self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#         # Candidate (Current Memory Content) GCN layer
#         self.conv_h = nn.GraphConv(self.in_channels,self.out_channels)

#         # Candidate linear layer
#         self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
#         return H

#     def _calculate_update_gate(self, graph, X, edge_weight, H):
#         Z = torch.cat((self.conv_z(graph, X, edge_weight=edge_weight), H), axis=1) # axis values need to be checked
#         Z = self.linear_z(Z)
#         Z = torch.sigmoid(Z)
#         return Z

#     def _calculate_reset_gate(self, graph, X, edge_weight, H):
#         R = torch.cat((self.conv_r(graph, X, edge_weight=edge_weight), H), axis=1) # axis values need to be checked
#         R = self.linear_r(R)
#         R = torch.sigmoid(R)
#         return R

#     def _calculate_candidate_state(self, graph, X, edge_weight, H, R):
#         H_tilde = torch.cat((self.conv_h(graph, X, edge_weight=edge_weight), H * R), axis=1) # axis values need to be checked
#         H_tilde = self.linear_h(H_tilde)
#         H_tilde = torch.tanh(H_tilde)
#         return H_tilde

#     def _calculate_hidden_state(self, Z, H, H_tilde):
#         H = Z * H + (1 - Z) * H_tilde
#         return H

#     def forward(
#         self,
#         graph: dgl.DGLGraph,
#         X: torch.Tensor,
#         edge_weight: torch.FloatTensor = None,
#         H: torch.FloatTensor = None,
#     ) -> torch.FloatTensor:
#         """
#         Making a forward pass. If edge weights are not present the forward pass
#         defaults to an unweighted graph. If the hidden state matrix is not present
#         when the forward pass is called it is initialized with zeros.

#         Arg types:
#             * **graph** *(DGL DGLGraph)* - Graph object.
#             * **X** *(PyTorch Float Tensor)* - Node features.
#             * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
#             * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

#         Return types:
#             * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
#         """

#         H = self._set_hidden_state(X, H)
#         Z = self._calculate_update_gate(graph, X, edge_weight, H)
#         R = self._calculate_reset_gate(graph, X, edge_weight, H)
#         H_tilde = self._calculate_candidate_state(graph, X, edge_weight, H, R)
#         H = self._calculate_hidden_state(Z, H, H_tilde)
#         return H

# class SeastarTGCN(torch.nn.Module):
#   def __init__(self, node_features):
#     super(SeastarTGCN, self).__init__()
#     self.temporal = SeastarTGCNCell(node_features, 32)
#     self.linear = torch.nn.Linear(32, 1)

#   def forward(self, graph, node_feat, edge_weight, hidden_state):
#     h = self.temporal(graph, node_feat, edge_weight, hidden_state)
#     y = F.relu(h)
#     y = self.linear(y)
#     return y, h