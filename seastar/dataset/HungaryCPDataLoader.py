import os
import json
from rich.console import Console
import numpy as np
console = Console()

class HungaryCPDataLoader:
    def __init__(self, cutoff_time, lags: int = 4, verbose: bool = False, for_seastar: bool = False):
        self.name = "HungaryCP"
        self._local_path = f'../../dataset/HungaryCP/{self.name}.json'
        self._verbose = verbose
        self.for_seastar = for_seastar
        self.lags = lags
        
        self._load_dataset()
        self.total_timestamps = min(len(self._dataset["FX"]), cutoff_time)
        
        self._get_num_nodes()
        self._get_num_edges()
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        
    def _load_dataset(self):
        if os.path.exists(self._local_path):
            dataset_file = open(self._local_path)
            self._dataset = json.load(dataset_file)
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/{self.name}.json')
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
        
    def _get_num_edges(self):
        self.num_edges = len(self._dataset["edges"])
        
    def _get_edges(self):
        if self.for_seastar:
            self._edge_list = [(edge[0], edge[1]) for edge in self._dataset["edges"]]
        else:
            self._edge_list = np.array(self._dataset["edges"]).T
            
    def _get_edge_weights(self):
        self._edge_weights = np.ones(self.num_edges)
        
    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self._all_features = np.array([
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ])
        self._all_targets = np.array([
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ])
        
    def get_edges(self):
        return self._edge_list

    def get_edge_weights(self):
        return self._edge_weights

    def get_all_targets(self):
        return self._all_targets
    
    def get_all_features(self):
        return self._all_features