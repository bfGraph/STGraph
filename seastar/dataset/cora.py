import os
import json
import urllib.request
import time

import numpy as np

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console

console = Console()

class CoraDataset:
    def __init__(self, verbose:bool = False, split=0.75) -> None:
        self.name = "Cora"
        self.split = split
        self.num_nodes = 0
        self.num_edges = 0

        self._local_file_path = "cora.json"
        self._url_path = "https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/cora.json"
        self._verbose = verbose
        
        # TODO: Remove
        # self._is_static = True
        # self._is_temporal = False
        # self._is_dynamic = False

        self._load_dataset()
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attributes()

    def _load_dataset(self) -> None:
        if self._is_local_exists():
            # loading the dataset from the local folder
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset locally')
            with open(self._local_file_path) as dataset_json:
                self._dataset = json.load(dataset_json)
        else:
            # loading the dataset by downloading them online
            if self._verbose:
                console.log(f'Downloading [cyan]{self.name}[/cyan] dataset')
            self._dataset = json.loads(urllib.request.urlopen(self._url_path).read())

            # TODO: Fix local file loadup
            # saving the dataset dictionary as a JSON file in local
            # with open(self._local_file_path, "w") as fp:
            #     json.dump(self._dataset, fp)
            #     if self._verbose:
            #         console.log(
            #             f"Successfully dowloaded [cyan]{self.name}[/cyan] dataset to [green]{self._local_file_path}[/green]"
            #         )

    def _get_edge_info(self):
        edges = np.array(self._dataset["edges"])
        edge_list = []
        for i in range(len(edges)):
            edge = edges[i]
            edge_list.append((edge[0], edge[1]))
            
        self._edge_list = edge_list

    def _get_targets_and_features(self):
        # NOTE: Should we return the transpose?
        self._all_features = np.array(self._dataset["features"])
        self._all_targets = np.array(self._dataset["labels"]).T

    def get_edges(self) -> np.ndarray:
        return self._edge_list

    def get_all_features(self) -> np.ndarray:
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        return self._all_targets

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

    def _is_local_exists(self) -> bool:
        # TODO: Fix local path issue
        return True
        return os.path.exists(self._local_file_path)
    
c = CoraDataset(verbose=True)
inspect(c._edge_list)