import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
import snoop
import inspect
from dgl.nn import GATv2Conv

class DGL_TGATCell(torch.nn.Module):

    def __init__(
        self,
        g,
        in_channels: int,
        out_channels: int,
        num_heads: int,
    ):
        super(DGL_TGATCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.g = g

        # Update GCN Layer
        self.conv_z = GATv2Conv(self.in_channels,self.out_channels,num_heads=self.num_heads,negative_slope=0.2,residual=True,share_weights=True,bias=False)
        
        # Update linear layer
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Reset GCN layer
        self.conv_r = GATv2Conv(self.in_channels,self.out_channels,num_heads=self.num_heads,negative_slope=0.2,residual=True,share_weights=True,bias=False)
        
        # Reset linear layer
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Candidate (Current Memory Content) GCN layer
        self.conv_h = GATv2Conv(self.in_channels,self.out_channels,num_heads=self.num_heads,negative_slope=0.2,residual=True,share_weights=True,bias=False)

        # Candidate linear layer
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.num_heads, self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_weight, H):
        h = self.conv_z(self.g, X)
        Z = torch.cat((h, H), axis=2) # axis values need to be checked
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_weight, H):
        R = torch.cat((self.conv_r(self.g,X), H), axis=2) # axis values need to be checked
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_weight, H, R):
        H_tilde = torch.cat((self.conv_h(self.g,X), H * R), axis=2) # axis values need to be checked
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

class DGL_TGAT(torch.nn.Module):
  def __init__(self, g, node_features):
    super(DGL_TGAT, self).__init__()
    self.g = g
    self.temporal = DGL_TGATCell(self.g, node_features, 32, 1)
    self.linear = torch.nn.Linear(32, 1)

  def forward(self, node_feat, edge_weight, hidden_state):
    h = self.temporal(node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    return y, h