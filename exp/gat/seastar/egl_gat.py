"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import argparse

import torch as th
import dgl
from torch import nn

import dgl.backend as B

from seastar import CtxManager

# pylint: enable=W0235
class EglGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(EglGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(
            self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope
        self.res_fc = nn.Linear(
                    self._in_feats, num_heads * out_feats, bias=False)
        self.reset_parameters()
        self.activation = activation
        self.cm = CtxManager(dgl.backend.run_egl)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()
        h_dst = h_src = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # Vertex-centric implementation.
        #dgl_context = dgl.utils.to_dgl_context(feat.device)
        #graph = graph._graph.get_immutable_gidx(dgl_context)
        #@self.cm.zoomIn(nspace=[self, th])
        #def nb_forward(v):
        #    coeff = [th.exp(self.leaky_relu(nb.el + v.er)) for nb in v.innbs]
        #    s = sum(coeff)
        #    alpha = [c/s for c in coeff]
        #    feat_src = [nb.feat_src for nb in v.innbs]
        #    return sum([alpha[i] * feat_src[i] for i in range(len(feat_src))])
        #rst = nb_forward(g=graph, n_feats= {'el':el, 'er': er, 'feat_src':feat_src})
        # Invoke fused kernel in dgl-hack
        rst = B.fused_gat(graph, feat_src, el, er, self.negative_slope)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class EglGAT(nn.Module):
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
        super(EglGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(EglGATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(EglGATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(EglGATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
