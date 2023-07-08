import numpy as np
import torch
from rich import inspect
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import matplotlib.pyplot as plt

adj_mat = np.load('adj_mat.npy')
adj_mat = torch.from_numpy(adj_mat)

edge_indices, values = dense_to_sparse(adj_mat)
edge_indices = edge_indices.numpy()
values = values.numpy()

edges = edge_indices
edge_weights = values
edge_list = []


for edge_index in range(len(edges[0])):
    edge_list.append((edges[0][edge_index], edges[1][edge_index]))

X = np.load("node_values.npy")


X = np.load("node_values.npy").transpose((1, 2, 0))
X = X.astype(np.float32)

# Normalise as in DCRNN paper (via Z-Score Method)
means = np.mean(X, axis=(0, 2))
X = X - means.reshape(1, -1, 1)
stds = np.std(X, axis=(0, 2))
X = X / stds.reshape(1, -1, 1)

X = torch.from_numpy(X)

inspect(X)
quit()

num_timesteps_in = 12
num_timesteps_out = 12

indices = [
    (i, i + (num_timesteps_in + num_timesteps_out))
    for i in range(X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
]

# Generate observations
features, target = [], []
for i, j in indices:
    features.append((X[:, :, i : i + num_timesteps_in]).numpy())
    target.append((X[:, 0, i + num_timesteps_in : j]).numpy())
    
inspect(len(features))
inspect(len(target))