import os
import json
from rich.console import Console
import numpy as np
console = Console()
import torch

from rich import inspect

class METRLADataLoader:
    def __init__(self ,num_timesteps_in:int = 12, num_timesteps_out:int = 12,verbose: bool = False, for_seastar: bool = False):
        self.name = "METRLA"
        self._local_path = f'../../dataset/{self.name}/METRLA.json'
        self._verbose = verbose
        self.for_seastar = for_seastar
        
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out
        
        self._load_dataset()
        self.total_timestamps = 10000

        self._get_num_nodes()
        self._get_edges()
        self._get_targets_and_features()
        
    def _load_dataset(self):
        # loading the dataset locally
        if os.path.exists(self._local_path):
            dataset_file = open(self._local_path)
            self._dataset = json.load(dataset_file)
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/')
        else:
            console.log(f'Failed to find [cyan]{self.name}[/cyan] dataset from dataset')
            quit()   
            
    def _get_num_nodes(self):
        node_set = set()
        max_node_id = 0
        for edge in self._dataset["edges"]:
            node_set.add(edge[0])
            node_set.add(edge[1])
            max_node_id = max(max_node_id, edge[0], edge[1])
        
        assert max_node_id == len(node_set) - 1, "Node ID labelling is not continuous"
        self.num_nodes = len(node_set)
            
    def _get_edges(self):
        if self.for_seastar:
            self._edge_list = [(edge[0], edge[1]) for edge in self._dataset["edges"]]
        else:
            self._edge_list = np.array(self._dataset["edges"]).T     
            
    # TODO: We are sorting the edge weights accordingly, but are we doing
    # the same for edges in the edge list
    def _get_edge_weights(self):
        if self.for_seastar:
            edges = self._dataset["edges"]
            edge_weights = self._dataset["weights"]
            comb_edge_list = [(edges[i][0], edges[i][1], edge_weights[i]) for i in range(len(edges))]
            comb_edge_list.sort(key=lambda x: (x[1], x[0]))
            self._edge_weights = np.array([edge_det[2] for edge_det in comb_edge_list])
        else:
            self._edge_weights = np.array(self._dataset["weights"])   
       
    def _get_targets_and_features(self):
        X = []
        
        for timestamp in range(self._dataset['time_periods']):
            if timestamp < self.total_timestamps:
                X.append(self._dataset[str(timestamp)])
            
        X = np.array(X)
        X = X.transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        X = torch.from_numpy(X)
            
        indices = [
            (i, i + (self.num_timesteps_in + self.num_timesteps_out))
            for i in range(X.shape[2] - (self.num_timesteps_in + self.num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((X[:, :, i : i + self.num_timesteps_in]).numpy())
            target.append((X[:, 0, i + self.num_timesteps_in : j]).numpy())

        self._all_features = np.array(features)
        self._all_targets = np.array(target)
        
    def get_edges(self):
        return self._edge_list
    
    def get_edge_weights(self):
        return self._edge_weights
    
    def get_all_targets(self):
        return self._all_targets
    
    def get_all_features(self):
        return self._all_features