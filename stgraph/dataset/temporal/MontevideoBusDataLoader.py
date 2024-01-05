import numpy as np

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class MontevideoBusDataLoader(STGraphTemporalDataset):
    def __init__(
        self,
        verbose: bool = False,
        url: str = None,
        lags: int = 4,
        cutoff_time: int = None,
        redownload: bool = False,
    ) -> None:
        r"""A dataset of inflow passenger at bus stop level from Montevideo city.

        This dataset compiles hourly passenger inflow data for 11 key bus lines
        in Montevideo, Uruguay, during October 2020. Focused on routes to the city
        center, it encompasses bus stop vertices, interlinked by edges representing
        connections with weights indicating road distances. The target variable
        is passenger inflow, sourced from diverse data outlets within Montevideo's
        Metropolitan Transportation System (STM).

        This class provides functionality for loading, processing, and accessing the
        Montevideo Bus dataset for use in deep learning tasks such as passenger inflow prediction.

        .. list-table:: gdata
            :widths: 33 33 33
            :header-rows: 1

            * - num_nodes
              - num_edges
              - total_timestamps
            * - 675
              - 690
              - 744

        Example
        -------

        .. code-block:: python

            from stgraph.dataset import MontevideoBusDataLoader

            monte = MontevideoBusDataLoader(verbose=True)
            num_nodes = monte.gdata["num_nodes"]
            num_edges = monte.gdata["num_edges"]
            total_timestamps = monte.gdata["total_timestamps"]

            edge_list = monte.get_edges()
            edge_weights = monte.get_edge_weights()
            feats = monte.get_all_features()
            targets = monte.get_all_targets()

        Parameters
        ----------

        verbose : bool, optional
            Flag to control whether to display verbose info (default is False)
        url : str, optional
            The URL from where the dataset is downloaded online (default is None)
        lags : int, optional
            The number of time lags (default is 4)
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
        _lags : int
            The number of time lags
        _cutoff_time : int
            The cutoff timestamp for the temporal dataset
        _edge_list : list
            The edge list of the graph dataset
        _edge_weights : numpy.ndarray
            Numpy array of the edge weights
        _all_targets : numpy.ndarray
            Numpy array of the node target value
        _all_features : numpy.ndarray
            Numpy array of the node feature value
        """

        super().__init__()

        if type(lags) != int:
            raise TypeError("lags must be of type int")
        if lags < 0:
            raise ValueError("lags must be a positive integer")

        if cutoff_time != None and type(cutoff_time) != int:
            raise TypeError("cutoff_time must be of type int")
        if cutoff_time != None and cutoff_time < 0:
            raise ValueError("cutoff_time must be a positive integer")
        if cutoff_time != None and cutoff_time <= lags:
            raise ValueError("cutoff_time must be greater than lags")

        self.name = "Montevideo_Bus"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/montevideobus.json"
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
        edges = self._dataset["edges"]
        edge_weights = self._dataset["weights"]
        comb_edge_list = [
            (edges[i][0], edges[i][1], edge_weights[i]) for i in range(len(edges))
        ]
        comb_edge_list.sort(key=lambda x: (x[1], x[0]))
        self._edge_weights = np.array([edge_det[2] for edge_det in comb_edge_list])

    def _set_features(self):
        r"""Calculates and sets the feature attributes"""
        features = []

        for node in self._dataset["nodes"]:
            X = node.get("X")
            for feature_var in ["y"]:
                features.append(
                    np.array(X.get(feature_var)[: self.gdata["total_timestamps"]])
                )

        stacked_features = np.stack(features).T
        standardized_features = (
            stacked_features - np.mean(stacked_features, axis=0)
        ) / np.std(stacked_features, axis=0)

        self._all_features = np.array(
            [
                standardized_features[i : i + self._lags, :].T
                for i in range(len(standardized_features) - self._lags)
            ]
        )

    def _set_targets(self):
        r"""Calculates and sets the target attributes"""
        targets = []
        for node in self._dataset["nodes"]:
            y = node.get("y")[: self.gdata["total_timestamps"]]
            targets.append(np.array(y))

        stacked_targets = np.stack(targets).T
        standardized_targets = (
            stacked_targets - np.mean(stacked_targets, axis=0)
        ) / np.std(stacked_targets, axis=0)

        self._all_targets = np.array(
            [
                standardized_targets[i + self._lags, :].T
                for i in range(len(standardized_targets) - self._lags)
            ]
        )

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
