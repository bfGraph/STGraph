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

class SeastarGATLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(SeastarGATLayer, self).__init__()
        self.g = g
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(
            self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope
        self.res_fc = nn.Linear(
                    self._in_feats, num_heads * out_feats, bias=False)
        self.reset_parameters()
        self.activation = activation
        self.cm = CtxManager(run_egl)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, feat):
        h_dst = h_src = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # Vertex-centric implementation.
        #dgl_context = dgl.utils.to_dgl_context(feat.device)
        #graph = graph._graph.get_immutable_gidx(dgl_context)
        @self.cm.zoomIn(nspace=[self, torch])
        def nb_forward(v):
           coeff = [torch.exp(self.leaky_relu(nb.el + v.er)) for nb in v.innbs]
           s = sum(coeff)
           alpha = [c/s for c in coeff]
           feat_src = [nb.feat_src for nb in v.innbs]
           return sum([alpha[i] * feat_src[i] for i in range(len(feat_src))])
        rst = nb_forward(g=self.g, n_feats= {'el':el, 'er': er, 'feat_src':feat_src})
        # Invoke fused kernel in dgl-hack
        # rst = B.fused_gat(graph, feat_src, el, er, self.negative_slope)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class SeastarTGATCell(torch.nn.Module):

    def __init__(
        self,
        g,
        in_channels: int,
        out_channels: int,
        num_heads: int,
    ):
        super(SeastarTGATCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.g = g

        # Update GCN Layer
        self.conv_z = SeastarGATLayer(self.g,self.in_channels,self.out_channels, self.num_heads)
        
        # Update linear layer
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Reset GCN layer
        self.conv_r = SeastarGATLayer(self.g,self.in_channels,self.out_channels, self.num_heads)
        
        # Reset linear layer
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Candidate (Current Memory Content) GCN layer
        self.conv_h = SeastarGATLayer(self.g, self.in_channels,self.out_channels, self.num_heads)

        # Candidate linear layer
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.num_heads, self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_weight, H):
        h = self.conv_z(X)
        Z = torch.cat((h, H), axis=2) # axis values need to be checked
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_weight, H):
        R = torch.cat((self.conv_r(X), H), axis=2) # axis values need to be checked
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_weight, H, R):
        H_tilde = torch.cat((self.conv_h(X), H * R), axis=2) # axis values need to be checked
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

class SeastarTGAT(torch.nn.Module):
  def __init__(self, g, node_features):
    super(SeastarTGAT, self).__init__()
    self.g = g
    self.temporal = SeastarTGATCell(self.g, node_features, 32, 1)
    self.linear = torch.nn.Linear(32, 1)

  def forward(self, node_feat, edge_weight, hidden_state):
    h = self.temporal(node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    return y, h