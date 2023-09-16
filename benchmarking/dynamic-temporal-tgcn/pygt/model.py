import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class PyGT_TGCN(torch.nn.Module):
  def __init__(self, node_features, num_hidden_units):
    super(PyGT_TGCN, self).__init__()
    self.temporal = TGCN(node_features, num_hidden_units, add_self_loops=False)
    self.linear = torch.nn.Linear(num_hidden_units, node_features)

  def forward(self, g, node_feat, edge_weight, hidden_state):
    h = self.temporal(g, node_feat, edge_weight, hidden_state)
    y = F.relu(h)
    y = self.linear(y)
    return y, h
  
  # product of a pair of nodes on each edge
  def decode(self, z, edge_label_index):
    return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
        dim=-1
    )
