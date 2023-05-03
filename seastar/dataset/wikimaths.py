import os
import json
import urllib.request

import numpy as np

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console

console = Console()


class WikiMaths:
    def __init__(self, verbose: bool = False, lags: int = 8, split=0.75) -> None:
        self.name = "WikiMaths"
        self.lags = lags
        self.split = split

        self._local_file_path = "wikimaths.json"
        self._url_path = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json"
        self._verbose = verbose
        self._is_static = False
        self._is_temporal = True
        self._is_dynamic = False

        self._load_dataset()
        self.total_timestamps = self._dataset["time_periods"]
        self._get_edge_info()
        self._get_node_info()
        self._get_targets_and_features()

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

    def _get_node_info(self):
        self._node_ids = self._dataset["node_ids"]

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

    def get_edges(self) -> np.ndarray:
        return self._edge_list

    def get_edge_weights(self) -> np.ndarray:
        return self._edge_weights

    def get_node_ids(self) -> dict:
        return self._node_ids

    def get_all_features(self) -> np.ndarray:
        return self._all_features

    def get_all_targets(self) -> np.ndarray:
        return self._all_targets

    def _is_local_exists(self) -> bool:
        return os.path.exists(self._local_file_path)