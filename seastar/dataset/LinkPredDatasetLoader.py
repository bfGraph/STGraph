import os
import json
import urllib.request
import time

import numpy as np
import random

from rich import inspect
from rich.pretty import pprint
from rich.progress import track
from rich.console import Console
import copy

console = Console()

# from rich.traceback import install
# install(show_locals=True)

class LinkPredDatasetLoader:
    def __init__(self, folder_name, dataset_name, max_num_nodes, verbose: bool = False, for_seastar= False) -> None:
        self.name = dataset_name

        self._max_num_nodes = max_num_nodes

        self._local_path = f'../../dataset/{folder_name}/{dataset_name}.json'
        self._verbose = verbose

        self._load_dataset()
        self.total_timestamps = self._dataset["time_periods"]
        self._dataset["time_periods"] = 4
        self.for_seastar = for_seastar
        
        self._get_edge_info()
        self._preprocess_pos_neg_edges()

    def _load_dataset(self) -> None:
        # loading the dataset by downloading them online
        # if self._verbose:
        #     console.log(f"Downloading [cyan]{self.name}[/cyan] dataset")
        
        # for online download
        if os.path.exists(self._local_path):
            dataset_file = open(self._local_path)
            self._dataset = json.load(dataset_file)
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/')
        else:
            console.log(f'Failed to find [cyan]{self.name}[/cyan] dataset from dataset')
            quit()
        
        # for local
        # dataset_file = open(self._local_path)
        # self._dataset = json.load(dataset_file)

    def _get_edge_info(self):
        # getting the edge_list and edge_weights
        edge_list = []
        updates = self._dataset["edge_mapping"]["edge_index"]

        updated_graph = updates["0"]["add"]
        edge_list.append(updated_graph)

        for time in range(1, self.total_timestamps):
            add_edges = updates[str(time)]["add"]
            delete_edges = updates[str(time)]["delete"]

            updated_graph = copy.deepcopy(updated_graph)
            updated_graph = updated_graph[len(delete_edges):] + add_edges
            edge_list.append(updated_graph)
        
        if self.for_seastar:
            self._edge_list = [[(edge[0], edge[1]) for edge in edge_lst_t] for edge_lst_t in edge_list]
        else:
            self._edge_list = [np.array(edge_lst_t).T for edge_lst_t in edge_list]
    
    def _preprocess_pos_neg_edges(self):
        updates = self._dataset["edge_mapping"]["edge_index"]

        pos_neg_edge_list = []
        pos_neg_edge_label_list = []

        for i in range(1, self.total_timestamps):
            pos_edges_tup = list(updates[str(i)]["add"])
            neg_edges_tup = list(updates[str(i)]["neg"])
            pos_neg_edge_list.append(pos_edges_tup + neg_edges_tup)
            pos_neg_edge_label_list.append([(edge[0], edge[1], 1) for edge in pos_edges_tup] + [(edge[0], edge[1], 0) for edge in neg_edges_tup])

        # if self.for_seastar:
        #     self._pos_neg_edge_list = [np.array(edge_list) for edge_list in pos_neg_edge_list]
        #     # pos_neg_edge_label_list.sort(key=lambda x: (x[1], x[0]))
        #     # self._pos_neg_edge_label_list = [np.array([edge[2] for edge in edge_list]) for edge_list in pos_neg_edge_label_list]
        #     self._pos_neg_edge_label_list = [np.array([edge[2] for edge in edge_list]) for edge_list in pos_neg_edge_label_list]
        # else:
        self._pos_neg_edge_list = [np.array(edge_list).T for edge_list in pos_neg_edge_list]
        self._pos_neg_edge_label_list = [np.array([edge[2] for edge in edge_list]) for edge_list in pos_neg_edge_label_list]

    def get_edges(self):
        return self._edge_list
    
    def get_pos_neg_edges(self):
        return self._pos_neg_edge_list, self._pos_neg_edge_label_list
