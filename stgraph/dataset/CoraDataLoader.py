"""Citation network consisting of scientific publications"""

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


class CoraDataLoader:
    r"""Citation network consisting of scientific publications

    The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
    The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued
    word vector indicating the absence/presence of the corresponding word from the dictionary.
    The dictionary consists of 1433 unique words.

    This class provides functionality for loading, processing, and accessing the Cora dataset
    for use in deep learning tasks such as graph-based node classification.

    .. list-table:: Dataset Stats
        :widths: 25 25 25 25
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 2708
          - 10556
          - 1433
          - 7

    Parameters
    ----------

    verbose : bool
        Indicate whether verbose output is needed while loading the dataset
    split : float
        Train to test split ratio

    Example
    -------

    .. code-block:: python

         from stgraph.dataset.CoraDataLoader import CoraDataLoader

         cora = CoraDataLoader()
         edge_list = cora.get_edges()
         all_feats = cora.get_all_features()
         all_targets = cora.get_all_targets()

    Attributes
    ----------
    name : str
        Name of the dataset
    num_nodes : int
        Number of nodes in the graph
    num_edges : int
        Number of edges in the graph
    _train_split : float
        Train split ratio of dataset
    _test_split : float
        Test split ratio of dataset
    _local_file_path : str
        Path to locally downloaded dataset file
    _url_path : str
        URL to download the dataset online
    _verbose : bool
        Verbose output flag
    _train_mask : list
        Training mask for the dataset
    _test_mask : list
        Testing mask for the dataset
    """

    def __init__(self, verbose: bool = False, split=0.75) -> None:
        self.name = "Cora"
        self.num_nodes = 0
        self.num_edges = 0

        self._train_split = split
        self._test_split = 1 - split

        self._local_file_path = f"../../dataset/cora/cora.json"
        self._url_path = (
            "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/cora.json"
        )
        self._verbose = verbose

        self._load_dataset()
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attributes()

        self._train_mask = [0] * self.num_nodes
        self._test_mask = [0] * self.num_nodes

        self._get_mask_info()

    def _load_dataset(self) -> None:
        r"""Loads the dataset either locally or downloads it from online"""
        if self._is_local_exists():
            # loading the dataset from the local folder
            if self._verbose:
                console.log(f"Loading [cyan]{self.name}[/cyan] dataset locally")
            with open(self._local_file_path) as dataset_json:
                self._dataset = json.load(dataset_json)
        else:
            # loading the dataset by downloading them online
            if self._verbose:
                console.log(f"Downloading [cyan]{self.name}[/cyan] dataset")
            self._dataset = json.loads(urllib.request.urlopen(self._url_path).read())

    def _get_edge_info(self) -> None:
        r"""Retrieves edge information from the dataset."""
        edges = np.array(self._dataset["edges"])
        edge_list = []
        for i in range(len(edges)):
            edge = edges[i]
            edge_list.append((edge[0], edge[1]))

        self._edge_list = edge_list

    def _get_targets_and_features(self) -> None:
        r"""Retrieves target labels and features from the dataset."""
        self._all_features = np.array(self._dataset["features"])
        self._all_targets = np.array(self._dataset["labels"]).T

    def _get_mask_info(self) -> None:
        r"""Generates training and testing masks."""
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
        r"""Returns edge list of the graph."""
        return self._edge_list

    def get_all_features(self) -> np.ndarray:
        r"""Returns all features of the dataset."""
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        r"""Returns all target labels of the dataset."""
        return self._all_targets

    def get_train_mask(self) -> np.ndarray:
        r"""Returns the training mask."""
        return self._train_mask

    def get_test_mask(self) -> np.ndarray:
        r"""Returns the testing mask."""
        return self._test_mask

    def get_train_features(self) -> np.ndarray:
        r"""Returns the training features."""
        train_range = int(len(self._all_features) * self.split)
        return self._all_features[:train_range]

    def get_train_targets(self) -> np.ndarray:
        r"""Returns the training target labels."""
        train_range = int(len(self._all_targets) * self.split)
        return self._all_targets[:train_range]

    def get_test_features(self) -> np.ndarray:
        r"""Returns the testing features."""
        test_range = int(len(self._all_features) * self.split)
        return self._all_features[test_range:]

    def get_test_targets(self) -> np.ndarray:
        r"""Returns the testing target labels."""
        test_range = int(len(self._all_targets) * self.split)
        return self._all_targets[test_range:]

    def _get_graph_attributes(self) -> None:
        r"""Computes the number of nodes and edges in the graph."""
        node_set = set()
        for edge in self._edge_list:
            node_set.add(edge[0])
            node_set.add(edge[1])

        self.num_nodes = len(node_set)
        self.num_edges = len(self._edge_list)

    def _is_local_exists(self) -> bool:
        r"""Checks if the local dataset file exists."""
        return os.path.exists(self._local_file_path)
