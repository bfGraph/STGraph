import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import urllib
from tqdm import tqdm
import snoop
import os

from rich import inspect
from rich.pretty import pprint
from rich.traceback import install
install(show_locals=True)

from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
import pandas as pd

class EnglandCovidDatasetLoader(object):

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):

        if os.path.exists("../../dataset/england_covid/data.json"):
            with open('../../dataset/england_covid/data.json', 'r') as openfile:
                self._dataset = json.load(openfile)
        else:
            url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
            self._dataset = json.loads(urllib.request.urlopen(url).read())
            self._save_dataset()

        

    def _save_dataset(self):
        with open("../../dataset/england_covid/data.json", "w") as outfile:
            json.dump(self._dataset,outfile)

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_mapping"]["edge_weight"][str(time)])
            )

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["y"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def get_dataset(self, lags: int = 8):
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return self._edges, self._edge_weights, self.features, self.targets


class BitcoinOtcDataloader():
    def __init__(self):
        self._dataset = pd.read_csv('../../dataset/bitcoin/soc-sign-bitcoinotc.csv',header=0, names=['source', 'target', 'rating','time'])
        self._dataset.time = self._dataset.time / 10000000
        self._dataset.time = self._dataset.time.astype(int)

        lst = np.append(self._dataset.source.astype(int).unique(),self._dataset.target.astype(int).unique())
        lst = np.sort(np.unique(lst))
        
        id_dict = {}
        for i in range(len(lst)):
            id_dict[lst[i]] = i
        self.node_id_dict = id_dict
            
        self._dataset.source = self._dataset.source.apply(lambda x: self.node_id_dict[x])
        self._dataset.target = self._dataset.target.apply(lambda x: self.node_id_dict[x])
    
    def _get_edges(self):
        self._edges = []
        time_lst = self._dataset.time.unique()
        time_lst.sort()
        for time in time_lst:
            filtered_df = self._dataset.where(self._dataset.time == time).dropna()
            self._edges.append(
                np.array([filtered_df.source.astype(int).tolist(),filtered_df.target.astype(int).tolist()])
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in self._dataset.time.unique():
            filtered_df = self._dataset.where(self._dataset.time == time).dropna()
            self._edge_weights.append(
                np.array(filtered_df.rating.tolist())
            )

    def _get_targets_and_features(self):
        pass

    def get_dataset(self):
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        return self._edges, self._edge_weights, None, None

def presort_edge_weights(edges_lst, edge_weights_lst):
    '''
        Presorting edges according to (dest,src) since that is how eids are formed
        allowing forward and backward kernel to access edge weights
    '''
    final_edges_lst = []
    final_edge_weights_lst = []

    for i in range(len(edges_lst)):
        src_list = edges_lst[i][0]
        dst_list = edges_lst[i][1]
        weights = edge_weights_lst[i]

        edge_info_list = []
        sorted_edges_lst = []
        sorted_edge_weights_lst = []

        for j in range(len(weights)):
            edge_info = (src_list[j], dst_list[j], weights[j])
            edge_info_list.append(edge_info)

        # sorted_edge_info_list = sorted(edge_info_list, key=lambda element: (element[0], element[1]))

        # since it has to be sorted according to the reverse order
        sorted_edge_info_list = sorted(edge_info_list, key=lambda element: (element[1], element[0]))

        temp_src = []
        temp_dst = []

        for edge in sorted_edge_info_list:
            temp_src.append(edge[0])
            temp_dst.append(edge[1])
            sorted_edge_weights_lst.append(edge[2])

        sorted_edges_lst.append(temp_src)
        sorted_edges_lst.append(temp_dst)
        sorted_edges_lst = np.array(sorted_edges_lst)

        final_edges_lst.append(sorted_edges_lst)
        final_edge_weights_lst.append(np.array(sorted_edge_weights_lst))

def preprocess_graph_structure(edges):
    # inspect(edges)
    tmp_set = set()
    for i in range(len(edges)):
        for j in range(len(edges[i][0])):
            tmp_set.add(edges[i][0][j])
            tmp_set.add(edges[i][1][j])
    max_num_nodes = len(tmp_set)

    edge_dict = {}
    for i in range(len(edges)):
        edge_set = set()
        for j in range(len(edges[i][0])):
            edge_set.add((edges[i][0][j],edges[i][1][j]))
        edge_dict[str(i)] = edge_set
    
    edge_final_dict = {}
    edge_final_dict["0"] = {"add": list(edge_dict["0"]),"delete": []}
    for i in range(1,len(edges)):
        edge_final_dict[str(i)] = {"add": list(edge_dict[str(i)].difference(edge_dict[str(i-1)])), "delete": list(edge_dict[str(i-1)].difference(edge_dict[str(i)]))}
    
    return edge_final_dict, max_num_nodes

# GPU | CPU
def get_default_device():
    
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def to_default_device(data):
    
    if isinstance(data,(list,tuple)):
        return [to_default_device(x,get_default_device()) for x in data]
    
    return data.to(get_default_device(),non_blocking = True)

def gpma_version(args):

    print("😍 GPMA Version")
    
    # Data
    # dataset = EnglandCovidDatasetLoader()
    dataset = BitcoinOtcDataloader()

    edges_lst, edge_weights_lst, all_features, all_targets = dataset.get_dataset()
    presort_edge_weights(edges_lst, edge_weights_lst)
    train_graph_log_dict, train_max_num_nodes = preprocess_graph_structure(edges_lst)
    
    # prev = train_graph_log_dict["0"]
    # curr = train_graph_log_dict["1"]
    
    # inspect(edge_weights_lst)
    
    # for key, val in curr["delete"]:
    #     found = False
    #     for k1, v1 in prev["add"]:
    #         if k1 == key and v1 == val:
    #             found = True
    #     if not found:
    #         print("Error ({},{}) not found".format(key,val))
    
    t0 = time.time()
    G = GPMAGraph(train_graph_log_dict,train_max_num_nodes)
    t1 = time.time()
    print("Timestamp = Base   |   Time Taken = {:.5f}".format(t1-t0))
    
    a = time.time()
    for i in range(1,18):
        t0 = time.time()
        G.get_graph(i)
        t1 = time.time()
        print("Timestamp = {}   |   Time Taken = {:.5f}".format(i,t1-t0))
    b = time.time()
    
    print(f'\n⏱️ Time taken for forward propagation: {b-a}\n')
    
    print("\n### Reverse Graph\n")
    
    a = time.time()
    for i in reversed(range(1,18)):
        t0 = time.time()
        G.get_backward_graph(i)
        t1 = time.time()
        print("Timestamp = {}   |   Time Taken = {:.5f}".format(i,t1-t0))
    b = time.time()

    print(f'\n⏱️ Time taken for backward propagation: {b-a}\n')
    
def pcsr_version(args):
    print("🎉 PCSR Version")
    
    # Data
    # dataset = EnglandCovidDatasetLoader()
    dataset = BitcoinOtcDataloader()

    edges_lst, edge_weights_lst, all_features, all_targets = dataset.get_dataset()
    presort_edge_weights(edges_lst, edge_weights_lst)
    train_graph_log_dict, train_max_num_nodes = preprocess_graph_structure(edges_lst)
    
    t0 = time.time()
    G = PCSRGraph(train_graph_log_dict,train_max_num_nodes)
    t1 = time.time()
    print("Timestamp = Base   |   Time Taken = {:.5f}".format(t1-t0))
    
    
    a = time.time()
    for i in range(1,18):
        t0 = time.time()
        G.get_graph(i)
        t1 = time.time()
        print("Timestamp = {}   |   Time Taken = {:.5f}".format(i,t1-t0))
    b = time.time()
    
    print(f'\n⏱️ Time taken for forward propagation: {b-a}\n')
    
    print("\n### Reverse Graph\n")
    
    a = time.time()
    for i in reversed(range(1,18)):
        t0 = time.time()
        G.get_backward_graph(i)
        t1 = time.time()
        print("Timestamp = {}   |   Time Taken = {:.5f}".format(i,t1-t0))
    b = time.time()

    print(f'\n⏱️ Time taken for backward propagation: {b-a}\n')

gpma_version(None)
print("\n----------------------------------------------------------\n")
pcsr_version(None)