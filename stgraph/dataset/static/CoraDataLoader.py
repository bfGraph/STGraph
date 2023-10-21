import json
import urllib.request
import random

import numpy as np
from rich.console import Console

from stgraph.dataset.static.STGraphStaticDataset import STGraphStaticDataset


console = Console()


class CoraDataLoader(STGraphStaticDataset):
    def __init__(self, verbose=False, url=None, split=0.75) -> None:
        r"""Citation network consisting of scientific publications

        The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
        The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued
        word vector indicating the absence/presence of the corresponding word from the dictionary.
        The dictionary consists of 1433 unique words.

        This class provides functionality for loading, processing, and accessing the Cora dataset
        for use in deep learning tasks such as graph-based node classification.

        .. list-table:: gdata
            :widths: 25 25 25 25
            :header-rows: 1

            * - num_nodes
              - num_edges
              - num_feats
              - num_classes
            * - 2708
              - 10556
              - 1433
              - 7

        Example
        -------

        .. code-block:: python

            from stgraph.dataset import CoraDataLoader

            cora = CoraDataLoader()
            num_nodes = cora.gdata["num_nodes"]
            edge_list = cora.get_edges()

        Parameters
        ----------

        verbose : bool, optional
            Flag to control whether to display verbose info (default is False)
        url : str, optional
            The URL from where the dataset is downloaded online (default is None)
        split : float, optional
            Train to test split ratio (default is 0.75)

        Attributes
        ----------
        name : str
            The name of the dataset.
        _verbose : bool
            Flag to control whether to display verbose info.
        _train_mask : np.ndarray
            Binary mask for train data.
        _test_mask : np.ndarray
            Binary mask for test data.
        _train_split : float
            Split ratio for train data.
        _test_split : float
            Split ratio for test data.
        """
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
        r"""Process the Cora dataset.

        Calls private methods to extract edge list, node features, target classes
        and the train/test binary mask array.
        """
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attributes()
        self._get_mask_info()

    def _get_edge_info(self) -> None:
        r"""Extract edge information from the dataset"""
        edges = np.array(self._dataset["edges"])
        edge_list = []
        for i in range(len(edges)):
            edge = edges[i]
            edge_list.append((edge[0], edge[1]))

        self._edge_list = edge_list

    def _get_targets_and_features(self):
        r"""Extract targets and features from the dataset."""
        self._all_features = np.array(self._dataset["features"])
        self._all_targets = np.array(self._dataset["labels"]).T

    def _get_graph_attributes(self):
        r"""Calculates and stores graph meta data inside ``gdata``"""
        node_set = set()
        for edge in self._edge_list:
            node_set.add(edge[0])
            node_set.add(edge[1])

        self.gdata["num_nodes"] = len(node_set)
        self.gdata["num_edges"] = len(self._edge_list)
        self.gdata["num_feats"] = len(self._all_features[0])
        self.gdata["num_classes"] = len(set(self._all_targets))

    def _get_mask_info(self):
        r"""Generate train and test binary masks array"""
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
        r"""Get the edge list."""
        return self._edge_list

    def get_all_features(self) -> np.ndarray:
        r"""Get all features."""
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        r"""Get all targets."""
        return self._all_targets

    def get_train_mask(self):
        r"""Get the train mask."""
        return self._train_mask

    def get_test_mask(self):
        r"""Get the test mask."""
        return self._test_mask

    def get_train_features(self) -> np.ndarray:
        r"""Get train features."""
        train_range = int(len(self._all_features) * self.split)
        return self._all_features[:train_range]

    def get_train_targets(self) -> np.ndarray:
        r"""Get train targets."""
        train_range = int(len(self._all_targets) * self.split)
        return self._all_targets[:train_range]

    def get_test_features(self) -> np.ndarray:
        r"""Get test features."""
        test_range = int(len(self._all_features) * self.split)
        return self._all_features[test_range:]

    def get_test_targets(self) -> np.ndarray:
        r"""Get test targets."""
        test_range = int(len(self._all_targets) * self.split)
        return self._all_targets[test_range:]
