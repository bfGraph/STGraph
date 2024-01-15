"""Dynamic dataset tracking COVID-19 cases in England's NUTS3 regions"""

import numpy as np

from stgraph.dataset.dynamic.STGraphDynamicDataset import STGraphDynamicDataset


class EnglandCovidDataLoader(STGraphDynamicDataset):
    r"""Dynamic dataset tracking COVID-19 cases in England's NUTS3 regions

    This dataset captures the interplay between COVID-19 cases and mobility
    in England's NUTS3 regions from March 3rd to May 12th. It is a directed
    and weighted graph that offers daily case count and movement of people
    between each region through node and edge features respectively.

    This class provides functionality for loading, processing, and accessing the England
    Covid dataset for use in deep learning tasks such as predicting the COVID cases
    in a region.

    References
    ----------

    `PyTorch Geometric Temporal EnglandCovidDatasetLoader <https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/encovid.html#EnglandCovidDatasetLoader>`__

    .. list-table:: gdata
        :widths: 33 33 33
        :header-rows: 1

        * - max_num_nodes
          - max_num_edges
          - total_timestamps
        * - 129
          - 2158
          - 61

    Example
    -------

    .. code-block:: python

        from stgraph.dataset import EnglandCovidDataLoader

        eng_covid = EnglandCovidDataLoader(verbose=True)
        num_nodes_dict = eng_covid.gdata["num_nodes"]
        num_edges_dict = eng_covid.gdata["num_edges"]
        total_timestamps = eng_covid.gdata["total_timestamps"]
        max_num_nodes = eng_covid.gdata["max_num_nodes"]
        max_num_edges = eng_covid.gdata["max_num_edges"]

        edge_list = eng_covid.get_edges()
        edge_weights = eng_covid.get_edge_weights()
        feats = eng_covid.get_all_features()
        targets = eng_covid.get_all_targets()

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
    gdata : dict
        The graph meta data
    """

    def __init__(
        self,
        verbose: bool = False,
        url: str = None,
        lags: int = 8,
        cutoff_time: int = None,
        redownload: bool = False,
    ) -> None:
        super().__init__()

        self.name = "England_COVID"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
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
        self._set_targets_and_features()
        self._set_edge_info()
        self._presort_edge_weights()

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

    def _set_targets_and_features(self):
        r"""Calculates and sets the target and feature attributes"""
        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10**-10
        )

        self._all_features = [
            standardized_target[i : i + self._lags, :].T
            for i in range(self.gdata["total_timestamps"] - self._lags)
        ]
        self._all_targets = [
            standardized_target[i + self._lags, :].T
            for i in range(self.gdata["total_timestamps"] - self._lags)
        ]

    def _set_edge_info(self):
        r"""Sets edge info such as edge list and edge weights"""
        self._edge_list = []
        self._edge_weights = []

        for time in range(self.gdata["total_timestamps"]):
            time_edge_list = []
            time_edge_weights = []

            for edge in self._dataset["edge_mapping"]["edge_index"][str(time)]:
                time_edge_list.append((edge[0], edge[1]))

            for weight in self._dataset["edge_mapping"]["edge_weight"][str(time)]:
                time_edge_weights.append(weight)

            self._edge_list.append(time_edge_list)
            self._edge_weights.append(time_edge_weights)
            self.gdata["num_edges"][str(time)] = len(time_edge_list)
            self.gdata["num_nodes"][str(time)] = len(
                {node for edge in time_edge_list for node in edge}
            )

            self.gdata["max_num_nodes"] = max(
                self.gdata["max_num_nodes"], self.gdata["num_nodes"][str(time)]
            )

            self.gdata["max_num_edges"] = max(
                self.gdata["max_num_edges"], self.gdata["num_edges"][str(time)]
            )

    def _presort_edge_weights(self):
        r"""
        Presorting edges according to (dest,src) since that is how eids are formed
        allowing forward and backward kernel to access edge weights
        """
        final_edges_lst = []
        final_edge_weights_lst = []

        for i in range(len(self._edge_list)):
            src_list = [edge[0] for edge in self._edge_list[i]]
            dst_list = [edge[1] for edge in self._edge_list[i]]
            weights = self._edge_weights[i]

            edge_info_list = []
            sorted_edge_weights_lst = []

            for j in range(len(weights)):
                edge_info = (src_list[j], dst_list[j], weights[j])
                edge_info_list.append(edge_info)

            # since it has to be sorted according to the reverse order
            sorted_edge_info_list = sorted(
                edge_info_list, key=lambda element: (element[1], element[0])
            )

            time_edge = []

            for edge in sorted_edge_info_list:
                time_edge.append((edge[0], edge[1]))
                sorted_edge_weights_lst.append(edge[2])

            final_edges_lst.append(time_edge)
            final_edge_weights_lst.append(np.array(sorted_edge_weights_lst))

        self._edge_list = final_edges_lst
        self._edge_weights = final_edge_weights_lst

    def get_edges(self) -> list:
        r"""Returns the edge list"""
        return self._edge_list

    def get_edge_weights(self) -> list:
        r"""Returns the edge weights"""
        return self._edge_weights

    def get_all_features(self) -> list:
        r"""Returns the features for each timestamp"""
        return self._all_features

    def get_all_targets(self) -> list:
        r"""Returns the targets for each timestamp"""
        return self._all_targets
