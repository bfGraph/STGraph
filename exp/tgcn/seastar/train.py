import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import json
import urllib
from tqdm import tqdm
from tgcn import SeastarTGCN
import snoop


class WikiMathsDGLDatasetLoader(object):
    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):

        targets = []
        for time in range(self._dataset["time_periods"]):
            targets.append(np.array(self._dataset[str(time)]["y"]))
        stacked_target = np.stack(targets)
        standardized_target = (
            stacked_target - np.mean(stacked_target, axis=0)
        ) / np.std(stacked_target, axis=0)
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(len(targets) - self.lags)
        ]

    def get_dataset(self, lags: int = 8):
            """Returning the Wikipedia Vital Mathematics data iterator.

            Args types:
                * **lags** *(int)* - The number of time lags.
            Return types:
                * **dataset** *(StaticGraphTemporalSignal)* - The Wiki Maths dataset.
            """
            self.lags = lags # how many time stamps before prediction has to be made
            self._get_edges()
            self._get_edge_weights()
            self._get_targets_and_features()
            return self._edges, self._edge_weights, self.features, self.targets

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
    dataset = WikiMathsDGLDatasetLoader()
    edges, edge_weights, all_features, all_targets = dataset.get_dataset()

    G = dgl.graph((torch.from_numpy(edges[0]),torch.from_numpy(edges[1]))) # Graph object
    G = to_default_device(dgl.add_self_loop(G))

    edge_weights = torch.FloatTensor(edge_weights)
    edge_weights = to_default_device(torch.cat((edge_weights,torch.ones(G.number_of_nodes())),0))

    # Seastar expects inputs to be of format (edge_weight,1)
    edge_weights = torch.unsqueeze(edge_weights,1)
    all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))

    # normalization
    degs = G.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = to_default_device(norm)
    G.ndata['norm'] = norm.unsqueeze(1)

    # Hyperparameters
    train_test_split = 0.8

    # train_test_split
    train_features = all_features[:int(len(all_features) * train_test_split)]
    train_targets = all_targets[:int(len(all_targets) * train_test_split)]
    test_features = all_features[int(len(all_features) * train_test_split):]
    test_targets = all_targets[int(len(all_targets) * train_test_split):]

    # train_features = all_features[:2]
    # train_targets = all_targets[:2]
    # print("Features")
    # print(train_features.shape)
    # print(train_targets.shape)

    # model
    model = to_default_device(SeastarTGCN(G,8))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # metrics
    dur = []
    Used_memory = 0
    cuda = True

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
        for index in range(train_features.shape[0]):
            y_hat, hidden_state = model(train_features[index], edge_weights, hidden_state)
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

    predictions = []
    true_y = []

    for index in range(test_features.shape[0]):
        y_hat, hidden_state = model(test_features[index], edge_weights, hidden_state)
        cost = cost + torch.mean((y_hat-test_targets[index])**2)
        predictions.append(y_hat)
        true_y.append(test_targets[index])
    cost = cost / (index+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    #add_argument --dataset
    register_data_args(parser)

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)


    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
            help="number of training epochs")
    args = parser.parse_args()
    print(args)

    main(args)
