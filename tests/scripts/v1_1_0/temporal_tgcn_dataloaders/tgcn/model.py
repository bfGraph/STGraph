import torch
import torch.nn.functional as F
from stgraph.nn.pytorch.temporal.tgcn import TGCN


class STGraphTGCN(torch.nn.Module):
    def __init__(self, node_features, num_hidden_units, out_features):
        super(STGraphTGCN, self).__init__()
        self.temporal = TGCN(node_features, num_hidden_units)
        self.linear = torch.nn.Linear(num_hidden_units, node_features)
        self.linear2 = torch.nn.Linear(node_features, out_features)

    def forward(self, g, node_feat, edge_weight, hidden_state):
        h = self.temporal(g, node_feat, edge_weight, hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        y_out = self.linear2(y)
        return y_out, y, h
