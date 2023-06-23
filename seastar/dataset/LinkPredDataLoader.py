import os
import json
import numpy as np
from rich.console import Console

console = Console()

class LinkPredDataLoader:
    def __init__(self, folder_name, dataset_name, cutoff_time, verbose: bool = False, for_seastar= False):
        self.name = dataset_name
        self._local_path = f'../../dataset/{folder_name}/{dataset_name}.json'
        self._verbose = verbose
        self.for_seastar = for_seastar

        self._load_dataset()
        self._get_max_num_nodes()
        self.total_timestamps = min(self._dataset["time_periods"], cutoff_time)
        self._get_edge_info()
        self._preprocess_pos_neg_edges()

    def _load_dataset(self) -> None:
        if os.path.exists(self._local_path):
            dataset_file = open(self._local_path)
            self._dataset = json.load(dataset_file)
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/')
        else:
            console.log(f'Failed to find [cyan]{self.name}[/cyan] dataset from dataset')
            quit()
    
    def _get_max_num_nodes(self):
        node_set = set()
        max_node_id = 0

        for i in range(len(self._dataset["edge_mapping"]["edge_index"])):
            for edge in self._dataset["edge_mapping"]["edge_index"][str(i)]["add"]:
                node_set.add(edge[0])
                node_set.add(edge[1])
                max_node_id = max(max_node_id, edge[0], edge[1])
        
        assert max_node_id == len(node_set) - 1, "Node ID labelling is not continuous"
        self.max_num_nodes = len(node_set)

    def _get_edge_info(self):
        # getting the edge_list and edge_weights
        edge_list = []
        updates = self._dataset["edge_mapping"]["edge_index"]

        working_set = set([(edge[0], edge[1]) for edge in updates["0"]["add"]])
        edge_list.append(list(working_set))
        for time in range(1, self.total_timestamps):
            working_set = working_set.union(set([(edge[0], edge[1]) for edge in updates[str(time)]["add"]])).difference(set([(edge[0], edge[1]) for edge in updates[str(time)]["delete"]]))
            edge_list.append(list(working_set))
        
        if self.for_seastar:
            self._edge_list = edge_list
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

        self._pos_neg_edge_list = [np.array(edge_list).T for edge_list in pos_neg_edge_list]
        self._pos_neg_edge_label_list = [np.array([edge[2] for edge in edge_list]) for edge_list in pos_neg_edge_label_list]

    def get_edges(self):
        return self._edge_list
    
    def get_pos_neg_edges(self):
        return self._pos_neg_edge_list, self._pos_neg_edge_label_list