from torch import nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

class RGCNModel(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNModel, self).__init__()

        self.layer1 = RelGraphConv(in_dim,
                                 hidden_dim,
                                 num_rels,
                                 self_loop=False)
        
        self.layer2 = RelGraphConv(hidden_dim,
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