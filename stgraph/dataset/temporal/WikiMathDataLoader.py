import numpy as np

from stgraph.dataset.temporal.stgraph_temporal_dataset import STGraphTemporalDataset


class WikiMathDataLoader(STGraphTemporalDataset):
    def __init__(
        self,
        verbose: bool = False,
        url: str = None,
        lags: int = 8,
        cutoff_time: int = None,
        redownload: bool = False,
    ) -> None:
        r"""A dataset of vital mathematical articles sourced from Wikipedia.

        The graph dataset is static, with vertices representing Wikipedia pages and
        edges representing links. The graph is both directed and weighted, where the weights
        indicate the number of links originating from the source page connecting
        to the target page. The target is the daily user visits to the Wikipedia pages
        between March 16th 2019 and March 15th 2021 which results in 731 periods.

        This class provides functionality for loading, processing, and accessing the Hungary
        Chickenpox dataset for use in deep learning tasks such as County level case count prediction.

        .. list-table:: gdata
            :widths: 33 33 33
            :header-rows: 1

            * - num_nodes
              - num_edges
              - total_timestamps
            * - 1068
              - 27079
              - 731

        Example
        -------

        .. code-block:: python

            from stgraph.dataset import WikiMathDataLoader

            wiki = WikiMathDataLoader(verbose=True)
            num_nodes = wiki.gdata["num_nodes"]
            num_edges = wiki.gdata["num_edges"]
            total_timestamps = wiki.gdata["total_timestamps"]

            edge_list = wiki.get_edges()
            edge_weights = wiki.get_edge_weights()
            targets = wiki.get_all_targets()

        Parameters
        ----------

        verbose : bool, optional
            Flag to control whether to display verbose info (default is False)
        url : str, optional
            The URL from where the dataset is downloaded online (default is None)
        lags : int, optional
            The number of time lags (default is 8)
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

        # TODO: Add check for cutoff_time <= lags

        self.name = "WikiMath"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/wikivital_mathematics.json"
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
        self._set_targets()
        self._set_features()

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

    def _set_targets(self):
        r"""Calculates and sets the target attributes"""
        targets = []
        for time in range(self.gdata["total_timestamps"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(targets)
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10**-10
        )
        self._all_targets = np.array(
            [standardized_target[i, :].T for i in range(len(targets))]
        )

    def _set_features(self):
        # TODO:
        pass

    def get_edges(self):
        r"""Returns the edge list"""
        return self._edge_list

    def get_edge_weights(self):
        r"""Returns the edge weights"""
        return self._edge_weights

    def get_all_targets(self):
        r"""Returns the targets for each timestamp"""
        return self._all_targets
