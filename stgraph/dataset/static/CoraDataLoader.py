import json
import urllib.request
import random

import numpy as np
from rich.console import Console

from stgraph.dataset.static.STGraphStaticDataset import STGraphStaticDataset


console = Console()


class CoraDataLoader(STGraphStaticDataset):
    def __init__(self, verbose=False, url=None, split=0.75) -> None:
        super().__init__()

        self.name = "Cora"
        self._verbose = verbose
        self._train_mask = None
        self._test_mask = None
        self._train_split = split
        self._test_split = 1 - split

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/cora.json"
        else:
            self._url = url

        if self._has_dataset_cache():
            self._load_dataset()
        else:
            self._download_dataset()
            self._save_dataset()

        self._process_dataset()

    def _process_dataset(self) -> None:
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attributes()
        self._get_mask_info()

    def _get_edge_info(self) -> None:
        edges = np.array(self._dataset["edges"])
        edge_list = []
        for i in range(len(edges)):
            edge = edges[i]
            edge_list.append((edge[0], edge[1]))

        self._edge_list = edge_list

    def _get_targets_and_features(self):
        self._all_features = np.array(self._dataset["features"])
        self._all_targets = np.array(self._dataset["labels"]).T

    def _get_graph_attributes(self):
        node_set = set()
        for edge in self._edge_list:
            node_set.add(edge[0])
            node_set.add(edge[1])

        self.gdata["num_nodes"] = len(node_set)
        self.gdata["num_edges"] = len(self._edge_list)

    def _get_mask_info(self):
        self._train_mask = [0] * self.gdata["num_nodes"]
        self._test_mask = [0] * self.gdata["num_nodes"]

        train_len = int(self.gdata["num_nodes"] * self._train_split)

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
