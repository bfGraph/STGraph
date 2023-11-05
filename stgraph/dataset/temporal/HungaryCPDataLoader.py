import numpy as np

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class HungaryCPDataLoader(STGraphTemporalDataset):
    def __init__(self, verbose=False, url=None, lags=4, cutoff_time=None) -> None:
        r"""County level chicken pox cases in Hungary

        This dataset comprises information on weekly occurrences of chickenpox
        in Hungary from 2005 to 2015. The graph structure is static with nodes
        representing the counties and edges are neighbourhoods between them.
        Vertex features are lagged weekly counts of the chickenpox cases.

        This class provides functionality for loading, processing, and accessing the Hungary
        Chickenpox dataset for use in deep learning tasks such as County level case count prediction.

        .. list-table:: gdata
            :widths: 33 33 33
            :header-rows: 1

            * - num_nodes
              - num_edges
              - total_timestamps
            * - 20
              - 102
              - 521

        Example
        -------

        .. code-block:: python

            from stgraph.dataset import HungaryCPDataLoader

            hungary = HungaryCPDataLoader(verbose=True)
            num_nodes = hungary.gdata["num_nodes"]
            edge_list = hungary.get_edges()

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

        assert lags > 0, "lags should be a positive integer"
        assert type(lags) == int, "lags should be of type int"
        assert cutoff_time > 0, "cutoff_time should be a positive integer"
        assert type(cutoff_time) == int, "cutoff_time should be a positive integer"

        self.name = "Hungary Chickenpox"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/HungaryCP.json"
        else:
            self._url = url

        if self._has_dataset_cache():
            self._load_dataset()
        else:
            self._download_dataset()
            self._save_dataset()

        self._process_dataset()

    def _process_dataset(self) -> None:
        self._get_total_timestamps()
        self._get_num_nodes()
        self._get_num_edges()
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()

    def _get_total_timestamps(self) -> None:
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                len(self._dataset["FX"]), self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = len(self._dataset["FX"])

    def _get_num_nodes(self):
        node_set = set()
        max_node_id = 0
        for edge in self._dataset["edges"]:
            node_set.add(edge[0])
            node_set.add(edge[1])
            max_node_id = max(max_node_id, edge[0], edge[1])

        assert max_node_id == len(node_set) - 1, "Node ID labelling is not continuous"
        self.gdata["num_nodes"] = len(node_set)

    def _get_num_edges(self):
        self.gdata["num_edges"] = len(self._dataset["edges"])

    def _get_edges(self):
        self._edge_list = [(edge[0], edge[1]) for edge in self._dataset["edges"]]

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self.gdata["num_edges"])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self._all_targets = np.array(
            [stacked_target[i, :].T for i in range(stacked_target.shape[0])]
        )

    def get_edges(self):
        return self._edge_list

    def get_edge_weights(self):
        return self._edge_weights

    def get_all_targets(self):
        return self._all_targets
