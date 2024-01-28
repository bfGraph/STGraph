r"""Hourly energy output of windmills."""

from __future__ import annotations

import numpy as np

from stgraph.dataset.temporal.stgraph_temporal_dataset import STGraphTemporalDataset


class WindmillOutputDataLoader(STGraphTemporalDataset):
    r"""Hourly energy output of windmills.

    This class provides functionality for loading, processing, and accessing
    the Windmill output dataset for use in deep learning such as
    regression tasks.

    .. list-table:: gdata for Windmill Output Small
        :widths: 33 33 33
        :header-rows: 1

        * - num_nodes
          - num_edges
          - total_timestamps
        * - 11
          - 121
          - 17472

    .. list-table:: gdata for Windmill Output Medium
        :widths: 33 33 33
        :header-rows: 1

        * - num_nodes
          - num_edges
          - total_timestamps
        * - 26
          - 676
          - 17472

    .. list-table:: gdata for Windmill Output Large
        :widths: 33 33 33
        :header-rows: 1

        * - num_nodes
          - num_edges
          - total_timestamps
        * - 319
          - 101761
          - 17472

    Example
    -------

    .. code-block:: python

        from stgraph.dataset import WindmillOutputDataLoader

        wind_small = WindmillOutputDataLoader(verbose=True, size="small")
        num_nodes = wind_small.gdata["num_nodes"]
        num_edges = wind_small.gdata["num_edges"]
        total_timestamps = wind_small.gdata["total_timestamps"]

        edge_list = wind_small.get_edges()
        edge_weights = wind_small.get_edge_weights()
        targets = wind_small.get_all_targets()

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
    size : str, optional
        The dataset size among large, medium and small (default is large)
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

    def __init__(
        self: WindmillOutputDataLoader,
        verbose: bool = False,
        url: str | None = None,
        lags: int = 8,
        cutoff_time: int | None = None,
        size: str = "large",
        redownload: bool = False,
    ) -> None:
        r"""Hourly energy output of windmills."""
        super().__init__()

        if not isinstance(lags, int):
            raise TypeError("lags must be of type int")
        if lags < 0:
            raise ValueError("lags must be a positive integer")

        if cutoff_time is not None and not isinstance(cutoff_time, int):
            raise TypeError("cutoff_time must be of type int")
        if cutoff_time is not None and cutoff_time < 0:
            raise ValueError("cutoff_time must be a positive integer")

        if not isinstance(size, str):
            raise TypeError("size must be of type string")
        if size not in ["large", "medium", "small"]:
            raise ValueError(
                "size must take either of the following values : "
                "large, medium or small",
            )

        self.name = "WindMill_" + size
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time
        self._size = size

        size_urls = {
            "large": "https://graphmining.ai/temporal_datasets/windmill_output.json",
            "medium": "https://graphmining.ai/temporal_datasets/windmill_output_medium.json",
            "small": "https://graphmining.ai/temporal_datasets/windmill_output_small.json",
        }

        if not url:
            self._url = size_urls[self._size]
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

    def _process_dataset(self: WindmillOutputDataLoader) -> None:
        self._set_total_timestamps()
        self._set_num_nodes()
        self._set_num_edges()
        self._set_edges()
        self._set_edge_weights()
        self._set_targets()

    def _set_total_timestamps(self: WindmillOutputDataLoader) -> None:
        r"""Set the total timestamps present in the dataset.

        It sets the total timestamps present in the dataset into the
        gdata attribute dictionary. It is the minimum of the cutoff time
        choosen by the user and the total time periods present in the
        original dataset.
        """
        if self._cutoff_time is not None:
            self.gdata["total_timestamps"] = min(
                self._dataset["time_periods"],
                self._cutoff_time,
            )
        else:
            self.gdata["total_timestamps"] = self._dataset["time_periods"]

    def _set_num_nodes(self: WindmillOutputDataLoader) -> None:
        r"""Set the total number of nodes present in the dataset."""
        node_set = set()
        max_node_id = 0
        for edge in self._dataset["edges"]:
            node_set.add(edge[0])
            node_set.add(edge[1])
            max_node_id = max(max_node_id, edge[0], edge[1])

        if max_node_id != len(node_set) - 1:
            raise ValueError("Node ID labelling is not continuous")

        self.gdata["num_nodes"] = len(node_set)

    def _set_num_edges(self: WindmillOutputDataLoader) -> None:
        r"""Set the total number of edges present in the dataset."""
        self.gdata["num_edges"] = len(self._dataset["edges"])

    def _set_edges(self: WindmillOutputDataLoader) -> None:
        r"""Set the edge list of the dataset."""
        self._edge_list = [(edge[0], edge[1]) for edge in self._dataset["edges"]]

    def _set_edge_weights(self: WindmillOutputDataLoader) -> None:
        r"""Set the edge weights of the dataset."""
        edges = self._dataset["edges"]
        edge_weights = self._dataset["weights"]
        comb_edge_list = [
            (edges[i][0], edges[i][1], edge_weights[i]) for i in range(len(edges))
        ]
        comb_edge_list.sort(key=lambda x: (x[1], x[0]))
        self._edge_weights = np.array([edge_det[2] for edge_det in comb_edge_list])

    def _set_targets(self: WindmillOutputDataLoader) -> None:
        r"""Calculate and sets the target attributes."""
        stacked_target = np.stack(self._dataset["block"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10**-10
        )
        self._all_targets = [
            standardized_target[i, :].T for i in range(self.gdata["total_timestamps"])
        ]

    def _set_features(self: WindmillOutputDataLoader) -> None:
        pass

    def get_edges(self: WindmillOutputDataLoader) -> list:
        r"""Return the edge list."""
        return self._edge_list

    def get_edge_weights(self: WindmillOutputDataLoader) -> np.ndarray:
        r"""Return the edge weight."""
        return self._edge_weights

    def get_all_targets(self: WindmillOutputDataLoader) -> list:
        r"""Return the targets for each timestamp."""
        return self._all_targets
