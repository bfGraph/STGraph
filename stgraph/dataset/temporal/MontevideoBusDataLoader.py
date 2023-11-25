import torch
import numpy as np

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class MontevideoBusDataLoader(STGraphTemporalDataset):
    def __init__(self, verbose=False, url=None, lags=4, cutoff_time=None) -> None:
        r"""A dataset of inflow passenger at bus stop level from Montevideo city."""

        super().__init__()

        assert lags > 0, "lags should be a positive integer"
        assert type(lags) == int, "lags should be of type int"
        assert cutoff_time > 0, "cutoff_time should be a positive integer"
        assert type(cutoff_time) == int, "cutoff_time should be a positive integer"

        self.name = "Montevideo Bus"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/montevideobus.json"
        else:
            self._url = url

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
        self._set_features()
        self._set_targets()

    def _set_total_timestamps(self) -> None:
        r"""Sets the total timestamps present in the dataset

        It sets the total timestamps present in the dataset into the
        gdata attribute dictionary. It is the minimum of the cutoff time
        choosen by the user and the total time periods present in the
        original dataset.
        """
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                len(self._dataset["nodes"][0]["y"]), self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = len(self._dataset["nodes"][0]["y"])

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

    def _set_features(self):
        r"""Calculates and sets the feature attributes"""
        features = []
        for node in self._dataset["nodes"]:
            X = node.get("X")
            for feature_var in ["y"]:
                features.append(np.array(X.get(feature_var)))

        stacked_features = np.stack(features).T
        standardized_features = (
            stacked_features - np.mean(stacked_features, axis=0)
        ) / np.std(stacked_features, axis=0)

        self._all_features = [
            standardized_features[i : i + self._lags, :].T
            for i in range(len(standardized_features) - self._lags)
        ]

    def _set_targets(self):
        r"""Calculates and sets the target attributes"""
        targets = []
        for node in self._dataset["nodes"]:
            y = node.get("y")
            targets.append(np.array(y))

        stacked_targets = np.stack(targets).T
        standardized_targets = (
            stacked_targets - np.mean(stacked_targets, axis=0)
        ) / np.std(stacked_targets, axis=0)

        self._all_targets = [
            standardized_targets[i + self._lags, :].T
            for i in range(len(standardized_targets) - self._lags)
        ]

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
