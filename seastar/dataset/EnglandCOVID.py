import os
import json
import urllib.request
import time

import numpy as np

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console

console = Console()

from rich.traceback import install
install(show_locals=True)

class EnglandCOVID:
    def __init__(self, verbose: bool = False, lags: int = 8, split=0.75) -> None:
        self.name = "EnglandCOVID"
        self.lags = lags
        self.split = split

        self._graph_attr = {}
        self._graph_updates = {}
        self._max_num_nodes = 0

        self._local_file_path = "england_covid.json"
        self._url_path = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
        self._verbose = verbose

        self._load_dataset()
        self.total_timestamps = self._dataset["time_periods"]
        self._get_edge_info()
        self._get_targets_and_features()
        self._get_graph_attr()
        self._presort_edge_weights()
        self._preprocess_graph_structure()

    def get_graph_data(self):
        return self._graph_updates, self._max_num_nodes

    def _load_dataset(self) -> None:
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

            # TODO: Fix local file loadup
            # saving the dataset dictionary as a JSON file in local
            # with open(self._local_file_path, "w") as fp:
            #     json.dump(self._dataset, fp)
            #     if self._verbose:
            #         console.log(
            #             f"Successfully dowloaded [cyan]WikiMath[/cyan] dataset to [green]{self._local_file_path}[/green]"
            #         )

    def _get_edge_info(self):
        # getting the edge_list and edge_weights
        self._edge_list = []
        self._edge_weights = []

        for time in range(self._dataset["time_periods"] - self.lags):
            time_edge_list = []
            time_edge_weights = []

            for edge in self._dataset["edge_mapping"]["edge_index"][str(time)]:
                time_edge_list.append((edge[0], edge[1]))

            for weight in self._dataset["edge_mapping"]["edge_weight"][str(time)]:
                time_edge_weights.append(weight)

            self._edge_list.append(time_edge_list)
            self._edge_weights.append(time_edge_weights)

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10**-10
        )

        self._all_features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self._all_targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def _get_graph_attr(self):
        # trying to calculate the num_nodes and num_edges
        # for each time stamp
        for time in range(len(self._edge_list)):
            node_set = set()
            edge_count = 0
            for edge in self._edge_list[time]:
                src = edge[0]
                dst = edge[1]
                edge_count += 1
                node_set.add(src)
                node_set.add(dst)
            node_count = len(node_set)
            self._graph_attr[str(time)] = (node_count, edge_count)

    def _presort_edge_weights(self):
        """
        Presorting edges according to (dest,src) since that is how eids are formed
        allowing forward and backward kernel to access edge weights
        """
        final_edges_lst = []
        final_edge_weights_lst = []

        # inspect(self._edge_list)
        # quit()

        for i in range(len(self._edge_list)):
            src_list = [edge[0] for edge in self._edge_list[i]]
            dst_list = [edge[1] for edge in self._edge_list[i]]
            # src_list = self._edge_list[i][0]
            # dst_list = self._edge_list[i][1]
            weights = self._edge_weights[i]

            edge_info_list = []
            sorted_edges_lst = []
            sorted_edge_weights_lst = []

            for j in range(len(weights)):
                edge_info = (src_list[j], dst_list[j], weights[j])
                edge_info_list.append(edge_info)

            # sorted_edge_info_list = sorted(edge_info_list, key=lambda element: (element[0], element[1]))

            # since it has to be sorted according to the reverse order
            sorted_edge_info_list = sorted(
                edge_info_list, key=lambda element: (element[1], element[0])
            )

            time_edge = []

            for edge in sorted_edge_info_list:
                time_edge.append((edge[0], edge[1]))
                sorted_edge_weights_lst.append(edge[2])

            # sorted_edges_lst.append(temp_src)
            # sorted_edges_lst.append(temp_dst)
            # sorted_edges_lst = np.array(sorted_edges_lst)

            final_edges_lst.append(time_edge)
            final_edge_weights_lst.append(np.array(sorted_edge_weights_lst))

        self._edge_list = final_edges_lst
        self._edge_weights = final_edge_weights_lst

    def _preprocess_graph_structure(self):
        tmp_set = set()
        for i in range(len(self._edge_list)):
            tmp_set = set()
            for j in range(len(self._edge_list[i])):
                tmp_set.add(self._edge_list[i][j][0])
                tmp_set.add(self._edge_list[i][j][1])
        self._max_num_nodes = len(tmp_set)

        edge_dict = {}
        for i in range(len(self._edge_list)):
            edge_set = set()
            for j in range(len(self._edge_list[i])):
                edge_set.add((self._edge_list[i][j][0], self._edge_list[i][j][1]))
            edge_dict[str(i)] = edge_set

        self._graph_updates = {}
        self._graph_updates["0"] = {
            "add": list(edge_dict["0"]),
            "delete": [],
            "num_nodes": self._graph_attr["0"][0],
            "num_edges": self._graph_attr["0"][1],
        }
        for i in range(1, len(self._edge_list)):
            self._graph_updates[str(i)] = {
                "add": list(edge_dict[str(i)].difference(edge_dict[str(i - 1)])),
                "delete": list(edge_dict[str(i - 1)].difference(edge_dict[str(i)])),
                "num_nodes": self._graph_attr[str(i)][0],
                "num_edges": self._graph_attr[str(i)][1],
            }

    def get_edges(self):
        return self._edge_list

    def get_edge_weights(self):
        return self._edge_weights

    def get_all_features(self):
        return self._all_features

    def get_all_targets(self):
        return self._all_targets

    def get_train_edge_list(self):
        train_range = int(len(self._edge_list) * self.split)
        self._edge_list[:train_range]
        
    def get_train_weights(self):
        train_range = int(len(self._edge_weights) * self.split)
        self._edge_weights[:train_range]

    def get_train_features(self):
        train_range = int(len(self._all_features) * self.split)
        return self._all_features[:train_range]

    def get_train_targets(self):
        train_range = int(len(self._all_targets) * self.split)
        return self._all_targets[:train_range]

    def get_test_edge_list(self):
        test_range = int(len(self._edge_list) * self.split)
        self._edge_list[test_range:]

    def get_test_weights(self):
        test_range = int(len(self._edge_weights) * self.split)
        self._edge_weights[test_range:]

    def get_test_features(self):
        test_range = int(len(self._all_features) * self.split)
        return self._all_features[test_range:]

    def get_test_targets(self):
        test_range = int(len(self._all_targets) * self.split)
        return self._all_targets[test_range:]

    def _is_local_exists(self) -> bool:
        # TODO: Fix local path issue
        return True
        return os.path.exists(self._local_file_path)
