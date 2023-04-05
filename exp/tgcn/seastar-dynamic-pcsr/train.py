import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import urllib
from tqdm import tqdm
from tgcn import SeastarTGCN
import snoop

from rich import inspect
# from rich.traceback import install
# install(show_locals=True)

from seastar_graph.seastar_graph import SeastarGraph

class EnglandCovidDatasetLoader(object):

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

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

def preprocess_graph_structure(edges):
    inspect(edges)
    tmp_set = set()
    for i in range(len(edges)):
        tmp_set = set()
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

def main(args):

    # Data
    dataset = EnglandCovidDatasetLoader()

    # edges_lst is a list of all edges of dynamic dataset, edge_weights_lst is the
    # corresponding edge weights.
    edges_lst, edge_weights_lst, all_features, all_targets = dataset.get_dataset()

    all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))

    # Hyperparameters
    train_test_split = 0.8

    # train_test_split for graph
    train_edges_lst = edges_lst[:int(len(edges_lst) * train_test_split)]
    test_edges_lst = edges_lst[int(len(edges_lst) * train_test_split):]
    train_edge_weights_lst = edge_weights_lst[:int(len(edge_weights_lst) * train_test_split)]
    test_edge_weights_lst = edge_weights_lst[int(len(edge_weights_lst) * train_test_split):]

    # train_test_split for features
    train_features = all_features[:int(len(all_features) * train_test_split)]
    train_targets = all_targets[:int(len(all_targets) * train_test_split)]
    test_features = all_features[int(len(all_features) * train_test_split):]
    test_targets = all_targets[int(len(all_targets) * train_test_split):]

    # model
    model = to_default_device(SeastarTGCN(8))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # metrics
    dur = []
    Used_memory = 0
    cuda = True

    train_graph_log_dict, train_max_num_nodes = preprocess_graph_structure(train_edges_lst)
    G = SeastarGraph(train_graph_log_dict,train_max_num_nodes)

    inspect(train_graph_log_dict)
    inspect(train_max_num_nodes)
    inspect(G)

    # train
    print("Training...\n")
    for epoch in tqdm(range(args.num_epochs)):

        model.train()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        cost = 0
        hidden_state = None
        optimizer.zero_grad()

        # dyn_graph_index is dynamic graph index
        for index in range(len(train_features)):  

            # normalization
            degs = torch.from_numpy(G.in_degrees())
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = to_default_device(norm)
            G.ndata['norm'] = norm.unsqueeze(1)

            print("🔴🔴 Attempting Forward prop")

            # forward propagation
            y_hat, hidden_state = model(G, train_features[index], None, hidden_state)
            cost = cost + torch.mean((y_hat-train_targets[index])**2)
        cost = cost / (index+1)

        now_mem = torch.cuda.max_memory_allocated(0)
        Used_memory = max(now_mem, Used_memory)

        cost.backward()
        optimizer.step()

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

        print('Epoch {:05d} | Time(s) {:.4f} | MSE {:.6f} | Used_Memory {:.6f} mb'.format(
            epoch, run_time_this_epoch, cost, (now_mem * 1.0 / (1024**2))
        ))

    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.mean(dur)))

    # evaluate
    print("Evaluating...\n")
    model.eval()
    cost = 0

    test_graph_log_dict, test_max_num_nodes = preprocess_graph_structure(test_edges_lst)
    G = SeastarGraph(test_graph_log_dict,test_max_num_nodes)

    predictions = []
    true_y = []
    hidden_state=None
    # dyn_graph_index is dynamic graph index
    for index in range(len(test_features)):
        # normalization
        degs = G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = to_default_device(norm)
        G.ndata['norm'] = norm.unsqueeze(1)
        
        # forward propagation
        y_hat, hidden_state = model(G, test_features[index], None, hidden_state)
        cost = cost + torch.mean((y_hat-test_targets[index])**2)
        predictions.append(y_hat)
        true_y.append(test_targets[index])

    cost = cost / (index+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)


    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
            help="number of training epochs")
    args = parser.parse_args()
    print(args)

    main(args)