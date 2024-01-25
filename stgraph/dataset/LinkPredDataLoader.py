import os
import json
import numpy as np
from rich.console import Console
import pickle

console = Console()

class LinkPredDataLoader:
    def __init__(self, folder_name, dataset_name, cutoff_time, verbose: bool = False, for_stgraph= False, for_stgraph_gpma=False):
        self.name = dataset_name
        self._local_path = f'../../dataset/{folder_name}/{dataset_name}'
        self._verbose = verbose
        self.for_stgraph = for_stgraph
        self.for_stgraph_gpma = for_stgraph_gpma
        self._edge_list = None

        self._load_dataset()
        self._get_max_num_nodes()
        self.total_timestamps = min(self._dataset["time_periods"], cutoff_time)

        if not self.for_stgraph_gpma:
            self._get_edge_info()

        self._preprocess_pos_neg_edges()

    def _load_dataset(self) -> None:
        if os.path.exists(f"{self._local_path}-metadata.json") and os.path.exists(f"{self._local_path}.npy") and os.path.exists(f"{self._local_path}-split.json"):

            # This includes metadata for splitting edges into snapshots
            dataset_file = open(f"{self.name}-metadata.json")
            self._dataset = json.load(dataset_file)
            dataset_file.close()

            if self.for_stgraph and not self.for_stgraph_gpma:
                dataset_file = open(f"{self.name}.pkl","rb")
                self._edges = pickle.load(dataset_file)
                dataset_file.close()
            elif not self.for_stgraph:
                self._edges = np.load(f"{self.name}.npy")
            
            # This includes edge list split as add, del and neg edges
            dataset_file = open(f"{self.name}-split.json")
            self.split_dataset = json.load(dataset_file)
            dataset_file.close()

            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/')
        else:
            console.log(f'Failed to find [cyan]{self.name}[/cyan] dataset from dataset')
            quit()
    
    def _get_max_num_nodes(self):
        self.max_num_nodes = self._dataset["max_num_nodes"]

    def _get_edge_info(self):
        # getting the edge_list and edge_weights
        edge_list = []
        edges = self._edges

        add_end_ptr = self._dataset["base"]
        delete_end_ptr = 0
        add_delta = self._dataset["add_delta"]
        delete_delta = self._dataset["delete_delta"]

        while add_end_ptr < len(edges):
            edge_list.append(edges[delete_end_ptr: add_end_ptr])
            delete_end_ptr += delete_delta
            add_end_ptr += add_delta
        
        if add_end_ptr - add_delta < len(edges):
            edge_list.append(edges[delete_end_ptr:])

        if self.for_stgraph:
            self._edge_list = edge_list
        else:
            self._edge_list = [edge_lst_t.T for edge_lst_t in edge_list]
    
    def _preprocess_pos_neg_edges(self):
        updates = self.split_dataset

        pos_neg_edge_list = []
        pos_neg_edge_label_list = []

        for i in range(1, self.total_timestamps):
            pos_edges_tup = list(updates[str(i)]["add"])
            neg_edges_tup = list(updates[str(i)]["neg"])
            pos_neg_edge_list.append(pos_edges_tup + neg_edges_tup)
            pos_neg_edge_label_list.append([ 1 for _ in pos_edges_tup] + [ 0 for _ in neg_edges_tup])

        self._pos_neg_edge_list = [np.array(edge_list).T for edge_list in pos_neg_edge_list]
        self._pos_neg_edge_label_list = [np.array(label_list_t) for label_list_t in pos_neg_edge_label_list]

    def get_edges(self):
            return self._edge_list
    
    def get_snapshot_edges(self):
            return self.split_dataset
    
    def get_pos_neg_edges(self):
        return self._pos_neg_edge_list, self._pos_neg_edge_label_list