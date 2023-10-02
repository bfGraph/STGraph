import torch
from torch import nn
from stgraph.compiler.backend.pytorch.torch_callback import STGraphBackendTorch
from stgraph.compiler import Seastar

# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None):
        super(GATConv, self).__init__()
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
        
        self.activation = activation
        self.seastar = Seastar(STGraphBackendTorch())
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):

        h_dst = h_src = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        # Vertex-centric implementation.
        @self.seastar.compile(gnn_module=self)
        def nb_forward(v):
            embs = [nb.el + v.er for nb in v.innbs]
            coeff = [torch.exp(self.leaky_relu(emb - max(embs))) for emb in embs]
            s = sum(coeff)
            alpha = [c/s for c in coeff]
            feat_src = [nb.feat_src for nb in v.innbs]
            return sum([alpha[i] * feat_src[i] for i in range(len(feat_src))])
        rst = nb_forward(g=graph, n_feats= {'el':el, 'er': er, 'feat_src':feat_src})

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
