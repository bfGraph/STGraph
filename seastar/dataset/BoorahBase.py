import os
import json
import urllib.request

import numpy as np

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console

console = Console()


class BoorahBase:
    def __init__(self, dataset_name, verbose: bool = False, lags: int = 8, split=0.75, for_seastar=False) -> None:
        self.name = dataset_name
        self.lags = lags
        self.split = split

        self._local_file_path = f"../../dataset/{dataset_name}/{dataset_name}.json"
        self._url_path = None
        self._verbose = verbose
        self._is_static = False
        self._is_temporal = True
        self._is_dynamic = False

        self._load_dataset()
        self.total_timestamps = self._dataset["time_periods"]
        self._get_targets_and_features()
        self._get_edge_info()
        
        if for_seastar:
            self._presort_edge_weights()

    def _load_dataset(self) -> None:
        if self._is_local_exists():
            # loading the dataset from the local folder
            if self._verbose:
                console.log("Loading [cyan]WikiMath[/cyan] dataset locally")
            with open(self._local_file_path) as dataset_json:
                self._dataset = json.load(dataset_json)
        else:
            # loading the dataset by downloading them online
            if self._verbose:
                console.log("Downloading [cyan]WikiMath[/cyan] dataset")
            self._dataset = json.loads(urllib.request.urlopen(self._url_path).read())

            # saving the dataset dictionary as a JSON file in local
            with open(self._local_file_path, "w") as fp:
                json.dump(self._dataset, fp)
                if self._verbose:
                    console.log(
                        f"Successfully dowloaded [cyan]WikiMath[/cyan] dataset to [green]{self._local_file_path}[/green]"
                    )
        
        

    def _get_edge_info(self):
        self._edge_list = np.array(self._dataset["edges"]).T
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        targets = []
        for time in range(self._dataset["time_periods"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))

        stacked_target = np.stack(targets)

        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)

        self._all_features = np.array(
            [
                standardized_target[i : i + self.lags, :].T
                for i in range(len(targets) - self.lags)
            ]
        )
        self._all_targets = np.array(
            [
                standardized_target[i + self.lags, :].T
                for i in range(len(targets) - self.lags)
            ]
        )
        
    def _presort_edge_weights(self):
        """
        Presorting edges according to (dest,src) since that is how eids are formed
        allowing forward and backward kernel to access edge weights
        """
        
        final_edge_list = []
        sorted_edge_weights_lst = []
        
        edge_info_list = [(self._edge_list[0][i],self._edge_list[1][i],self._edge_weights[i]) for i in range(len(self._edge_list[0]))]

        # since it has to be sorted according to the (dst,src) order
        sorted_edge_info_list = sorted(
            edge_info_list, key=lambda element: (element[1], element[0])
        )

        for edge in sorted_edge_info_list:
            final_edge_list.append((edge[0], edge[1]))
            sorted_edge_weights_lst.append(edge[2])

        self._edge_list = final_edge_list
        self._edge_weights = sorted_edge_weights_lst

    def get_edges(self) -> np.ndarray:
        return self._edge_list

    def get_edge_weights(self) -> np.ndarray:
        return self._edge_weights

    def get_all_features(self) -> np.ndarray:
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        return self._all_targets

    def _is_local_exists(self) -> bool:
        return os.path.exists(self._local_file_path)
    