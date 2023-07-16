import os
import json
from rich.console import Console
import numpy as np
console = Console()

class MontevideoBusDataLoader:
    def __init__(self, folder_name, dataset_name, lags, cutoff_time, verbose: bool = False, for_seastar = False) -> None:
        self.name = dataset_name
        self._local_path = f'../../dataset/{folder_name}/{dataset_name}.json'
        self._verbose = verbose
        self.for_seastar = for_seastar
        self.lags = lags
        
        self._load_dataset()
        self.total_timestamps = min(len(self._dataset["nodes"][0]['y']), cutoff_time)
        
        self._get_num_nodes()
        self._get_num_edges()
        self._get_edges()
        self._get_edge_weights()
        self._get_targets()
        
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
        if self.for_seastar:
            edges = self._dataset["edges"]
            edge_weights = self._dataset["weights"]
            comb_edge_list = [(edges[i][0], edges[i][1], edge_weights[i]) for i in range(len(edges))]
            comb_edge_list.sort(key=lambda x: (x[1], x[0]))
            self._edge_weights = np.array([edge_det[2] for edge_det in comb_edge_list])
        else:
            self._edge_weights = np.array(self._dataset["weights"])
        
    def _get_targets(self, target_var: str = "y"):
        targets = []
        for node in self._dataset["nodes"]:
            y = node.get(target_var)
            targets.append(np.array(y))
        stacked_targets = np.stack(targets).T
        standardized_targets = (
            stacked_targets - np.mean(stacked_targets, axis=0)
        ) / np.std(stacked_targets, axis=0)
        self._all_targets = np.array([
            standardized_targets[i, :].T
            for i in range(len(standardized_targets))
        ])
        
    def get_edges(self):
        return self._edge_list

    def get_edge_weights(self):
        return self._edge_weights

    def get_all_targets(self):
        return self._all_targets