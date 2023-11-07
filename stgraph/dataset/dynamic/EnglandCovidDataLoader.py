import numpy as np

from stgraph.dataset.dynamic.STGraphDynamicDataset import STGraphDynamicDataset


class EnglandCovidDataLoader(STGraphDynamicDataset):
    def __init__(
        self,
        verbose: bool = False,
        url: str = None,
        lags: int = 8,
        cutoff_time: int = None,
    ) -> None:
        super().__init__()

        self.name = "England COVID"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
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
        self._get_targets_and_features()
        self._get_edge_info()
        self._presort_edge_weights()

    def _get_total_timestamps(self) -> None:
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                self._dataset["time_periods"], self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = self._dataset["time_periods"]

    def _get_targets_and_features(self):
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

    def _get_edge_info(self):
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

    def _presort_edge_weights(self):
        """
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

    def get_edges(self):
        return self._edge_list

    def get_edge_weights(self):
        return self._edge_weights

    def get_all_features(self):
        return self._all_features

    def get_all_targets(self):
        return self._all_targets
