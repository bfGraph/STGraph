import json
import numpy as np
import torch
from rich import inspect
from torch_geometric.utils import dense_to_sparse

# the final json object we need
metrla_json = {}

metrla_json["edges"] = []

adj_mat = np.load('adj_mat.npy')
adj_mat = torch.from_numpy(adj_mat)

edge_indices, values = dense_to_sparse(adj_mat)
edge_indices = edge_indices.numpy()
values = values.numpy()

edges = edge_indices
edge_weights = values

edge_list = []
edge_weight_list = []

for edge_index in range(len(edges[0])):
    edge_list.append([int(edges[0][edge_index]), int(edges[1][edge_index])])
    
for edge_weight in edge_weights:
    edge_weight_list.append(float(edge_weight))
    
metrla_json["edges"] = edge_list
metrla_json["weights"] = edge_weight_list
metrla_json["time_periods"] = 34272

node_features = np.load('node_values.npy')
inspect(node_features)

for time_period in range(len(node_features)):
    time_node_feat_list = []
    for node_feats in node_features[time_period]:
        time_node_feat_list.append(list(node_feats))
    metrla_json[str(time_period)] = time_node_feat_list

inspect(metrla_json)

with open('METRLA.json', 'w') as fp:
    json.dump(metrla_json, fp)