"""A traffic forecasting dataset based on Los Angeles Metropolitan traffic conditions"""

import torch
import numpy as np

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class METRLADataLoader(STGraphTemporalDataset):
    r"""A traffic forecasting dataset based on Los Angeles Metropolitan traffic conditions.

    A dataset for predicting traffic patterns in the Los Angeles Metropolitan area,
    comprising traffic data obtained from 207 loop detectors on highways in Los Angeles County.
    The dataset includes aggregated 5-minute interval readings spanning a four-month
    period from March 2012 to June 2012.

    This class provides functionality for loading, processing, and accessing the METRLA
    dataset for use in deep learning tasks such as traffic forecasting.

    .. list-table:: gdata
        :widths: 33 33 33
        :header-rows: 1

        * - num_nodes
          - num_edges
          - total_timestamps
        * - 207
          - 1722
          - 100

    Example
    -------

    .. code-block:: python

        from stgraph.dataset import METRLADataLoader

        metrla = METRLADataLoader(verbose=True)
        num_nodes = metrla.gdata["num_nodes"]
        num_edges = metrla.gdata["num_edges"]
        total_timestamps = metrla.gdata["total_timestamps"]

        edge_list = metrla.get_edges()
        edge_weights = metrla.get_edge_weights()
        feats = metrla.get_all_features()
        targets = metrla.get_all_targets()

    Parameters
    ----------

    verbose : bool, optional
        Flag to control whether to display verbose info (default is False)
    url : str, optional
        The URL from where the dataset is downloaded online (default is None)
    num_timesteps_in : int, optional
        The number of timesteps the sequence model sees (default is 12)
    num_timesteps_out : int, optional
        The number of timesteps the sequence model has to predict (default is 12)
    cutoff_time : int, optional
        The cutoff timestamp for the temporal dataset (default is None)
    redownload : bool, optional (default is False)
        Redownload the dataset online and save to cache

    Attributes
    ----------
    name : str
        The name of the dataset.
    _verbose : bool
        Flag to control whether to display verbose info.
    _num_timesteps_in : int
        The number of timesteps the sequence model sees
    _num_timesteps_out : int
        The number of timesteps the sequence model has to predict
    _cutoff_time : int
        The cutoff timestamp for the temporal dataset
    _edge_list : list
        The edge list of the graph dataset
    _edge_weights : numpy.ndarray
        Numpy array of the edge weights
    _all_features : numpy.ndarray
        Numpy array of the node feature value
    _all_targets : numpy.ndarray
        Numpy array of the node target value
    """

    def __init__(
        self,
        verbose: bool = True,
        url: str = None,
        num_timesteps_in: int = 12,
        num_timesteps_out: int = 12,
        cutoff_time: int = None,
        redownload: bool = False,
    ):
        super().__init__()

        if type(num_timesteps_in) != int:
            raise TypeError("num_timesteps_in must be of type int")
        if num_timesteps_in < 0:
            raise ValueError("num_timesteps_in must be a positive integer")

        if type(num_timesteps_out) != int:
            raise TypeError("num_timesteps_out must be of type int")
        if num_timesteps_out < 0:
            raise ValueError("num_timesteps_out must be a positive integer")

        if cutoff_time != None and type(cutoff_time) != int:
            raise TypeError("cutoff_time must be of type int")
        if cutoff_time != None and cutoff_time < 0:
            raise ValueError("cutoff_time must be a positive integer")

        self.name = "METRLA"
        self._verbose = verbose
        self._num_timesteps_in = num_timesteps_in
        self._num_timesteps_out = num_timesteps_out
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/METRLA.json"
        else:
            self._url = url

        if redownload and self._has_dataset_cache():
            self._delete_cached_dataset()

        if self._has_dataset_cache():
            self._load_dataset()
        else:
            self._download_dataset()
            self._save_dataset()

        self._process_dataset()

    def _process_dataset(self) -> None:
        self._set_total_timestamps()
        self._set_num_nodes()
        self._set_num_edges()
        self._set_edges()
        self._set_edge_weights()
        self._set_targets_and_features()

    def _set_total_timestamps(self) -> None:
        r"""Sets the total timestamps present in the dataset

        It sets the total timestamps present in the dataset into the
        gdata attribute dictionary. It is the minimum of the cutoff time
        choosen by the user and the total time periods present in the
        original dataset.
        """
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                self._dataset["time_periods"], self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = self._dataset["time_periods"]

    def _set_num_nodes(self):
        r"""Sets the total number of nodes present in the dataset"""
        node_set = set()
        max_node_id = 0
        for edge in self._dataset["edges"]:
            node_set.add(edge[0])
            node_set.add(edge[1])
            max_node_id = max(max_node_id, edge[0], edge[1])

        assert max_node_id == len(node_set) - 1, "Node ID labelling is not continuous"
        self.gdata["num_nodes"] = len(node_set)

    def _set_num_edges(self):
        r"""Sets the total number of edges present in the dataset"""
        self.gdata["num_edges"] = len(self._dataset["edges"])

    def _set_edges(self):
        r"""Sets the edge list of the dataset"""
        self._edge_list = [(edge[0], edge[1]) for edge in self._dataset["edges"]]

    def _set_edge_weights(self):
        r"""Sets the edge weights of the dataset"""
        edges = self._dataset["edges"]
        edge_weights = self._dataset["weights"]
        comb_edge_list = [
            (edges[i][0], edges[i][1], edge_weights[i]) for i in range(len(edges))
        ]
        comb_edge_list.sort(key=lambda x: (x[1], x[0]))
        self._edge_weights = np.array([edge_det[2] for edge_det in comb_edge_list])

    def _set_targets_and_features(self):
        r"""Calculates and sets the target and feature attributes"""
        X = []

        for timestamp in range(self.gdata["total_timestamps"]):
            X.append(self._dataset[str(timestamp)])

        X = np.array(X)
        X = X.transpose((1, 2, 0))
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        X = torch.from_numpy(X)

        indices = [
            (i, i + (self._num_timesteps_in + self._num_timesteps_out))
            for i in range(
                X.shape[2] - (self._num_timesteps_in + self._num_timesteps_out) + 1
            )
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((X[:, :, i : i + self._num_timesteps_in]).numpy())
            target.append((X[:, 0, i + self._num_timesteps_in : j]).numpy())

        self._all_features = np.array(features)
        self._all_targets = np.array(target)

    def get_edges(self):
        r"""Returns the edge list"""
        return self._edge_list

    def get_edge_weights(self):
        r"""Returns the edge weights"""
        return self._edge_weights

    def get_all_targets(self):
        r"""Returns the targets for each timestamp"""
        return self._all_targets

    def get_all_features(self):
        r"""Returns the features for each timestamp"""
        return self._all_features
