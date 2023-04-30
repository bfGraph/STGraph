import os
import json
import urllib.request
import time
import random

import numpy as np

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console

console = Console()

class CoraDataset:
    def __init__(self, verbose:bool = False, split=0.75) -> None:
        self.name = "Cora"
        self.num_nodes = 0
        self.num_edges = 0
        
        self._train_split = split
        self._test_split = 1-split

        self._url_path = "https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/cora.json"
        self._verbose = verbose

        self._load_dataset()
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attributes()
        
        self._train_mask = [0] * self.num_nodes
        self._test_mask = [0] * self.num_nodes
        
        self._get_mask_info()

    def _load_dataset(self) -> None:
        
        # loading the dataset by downloading them online
        if self._verbose:
            console.log(f'Downloading [cyan]{self.name}[/cyan] dataset')
        self._dataset = json.loads(urllib.request.urlopen(self._url_path).read())


    def _get_edge_info(self):
        edges = np.array(self._dataset["edges"])
        edge_list = []
        for i in range(len(edges)):
            edge = edges[i]
            edge_list.append((edge[0], edge[1]))
            
        self._edge_list = edge_list

    def _get_targets_and_features(self):
        self._all_features = np.array(self._dataset["features"])
        self._all_targets = np.array(self._dataset["labels"]).T

    def _get_mask_info(self):
        train_len = int(self.num_nodes * self._train_split)
        
        for i in range(0, train_len):
            self._train_mask[i] = 1
            
        random.shuffle(self._train_mask)
            
        for i in range(len(self._train_mask)):
            if self._train_mask[i] == 0:
                self._test_mask[i] = 1
            
        self._train_mask = np.array(self._train_mask)
        self._test_mask = np.array(self._test_mask)

    def get_edges(self) -> np.ndarray:
        return self._edge_list

    def get_all_features(self) -> np.ndarray:
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        return self._all_targets

    def get_train_mask(self):
        return self._train_mask
    
    def get_test_mask(self):
        return self._test_mask

    def get_train_features(self) -> np.ndarray:
        train_range = int(len(self._all_features) * self.split)
        return self._all_features[:train_range]

    def get_train_targets(self) -> np.ndarray:
        train_range = int(len(self._all_targets) * self.split)
        return self._all_targets[:train_range]

    def get_test_features(self) -> np.ndarray:
        test_range = int(len(self._all_features) * self.split)
        return self._all_features[test_range:]

    def get_test_targets(self) -> np.ndarray:
        test_range = int(len(self._all_targets) * self.split)
        return self._all_targets[test_range:]

    def _get_graph_attributes(self):
        node_set = set()
        for edge in self._edge_list:
            node_set.add(edge[0])
            node_set.add(edge[1])
            
        self.num_nodes = len(node_set)
        self.num_edges = len(self._edge_list)