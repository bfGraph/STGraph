import torch
import torch.nn.functional as F
from seastar.nn.pytorch.temporal.tgcn import TGCN


class SeastarTGCN(torch.nn.Module):
    def __init__(self, node_features, num_hidden_units, out_features):
        super(SeastarTGCN, self).__init__()
        self.temporal = TGCN(node_features, num_hidden_units)
        self.linear = torch.nn.Linear(num_hidden_units, out_features)

    def forward(self, g, node_feat, edge_weight, hidden_state):
        h = self.temporal(g, node_feat, edge_weight, hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h
