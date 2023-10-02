import torch
import torch.nn as nn
from stgraph.compiler import STGraph
from stgraph.compiler.backend.pytorch.torch_callback import STGraphBackendTorch

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.stgraph = STGraph(STGraphBackendTorch())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, h, edge_weight=None):
        h = torch.mm(h, self.weight)
        
        if edge_weight is None:
            @self.stgraph.compile(gnn_module=self)
            def nb_compute(v):
                h = sum([nb.h*nb.norm for nb in v.innbs])
                h = h * v.norm
                return h
            h = nb_compute(g=g, n_feats={'norm': g.get_ndata("norm"), 'h' : h})
        else:
            @self.stgraph.compile(gnn_module=self)
            def nb_compute(v):
                h = sum([nb_edge.src.norm * nb_edge.src.h * nb_edge.edge_weight for nb_edge in v.inedges])
                h = h * v.norm
                return h
            h = nb_compute(g=g, n_feats={'norm': g.get_ndata("norm"), 'h' : h}, e_feats={'edge_weight':edge_weight})

        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h
