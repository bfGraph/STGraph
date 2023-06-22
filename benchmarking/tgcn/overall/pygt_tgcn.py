import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class PyGT_TGCN(torch.nn.Module):
  def __init__(self, node_features):
    super(PyGT_TGCN, self).__init__()
    self.temporal = TGCN(node_features, 2*node_features)
    self.linear = torch.nn.Linear(2*node_features, node_features)

  def forward(self, g, node_feat, edge_weight, hidden_state):
    h = self.temporal(g, node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    
    return y, h