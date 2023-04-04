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
        for index in range(len(train_edges_lst)):
            edges = train_edges_lst[index]
            edge_weights = train_edge_weights_lst[index]

            G = dgl.graph((torch.from_numpy(edges[0]),torch.from_numpy(edges[1]))) # Graph object
            G = to_default_device(dgl.add_self_loop(G))

            edge_weights = torch.FloatTensor(edge_weights)
            edge_weights = to_default_device(torch.cat((edge_weights,torch.ones(G.number_of_nodes())),0))

            # Seastar expects inputs to be of format (edge_weight,1)
            edge_weights = torch.unsqueeze(edge_weights,1)
            
            # normalization
            degs = G.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = to_default_device(norm)
            G.ndata['norm'] = norm.unsqueeze(1)

            # forward propagation
            y_hat, hidden_state = model(G, train_features[index], edge_weights, hidden_state)
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
    hidden_state=None
    # dyn_graph_index is dynamic graph index
    for index in range(len(test_edges_lst)):
        edges = test_edges_lst[index]
        edge_weights = test_edge_weights_lst[index]

        G = dgl.graph((torch.from_numpy(edges[0]),torch.from_numpy(edges[1]))) # Graph object
        G = to_default_device(dgl.add_self_loop(G))

        edge_weights = torch.FloatTensor(edge_weights)
        edge_weights = to_default_device(torch.cat((edge_weights,torch.ones(G.number_of_nodes())),0))

        # Seastar expects inputs to be of format (edge_weight,1)
        edge_weights = torch.unsqueeze(edge_weights,1)
        
        # normalization
        degs = G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = to_default_device(norm)
        G.ndata['norm'] = norm.unsqueeze(1)

        # forward propagation
        y_hat, hidden_state = model(G, test_features[index], edge_weights, hidden_state)
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
    parser.add_argument("--num_epochs", type=int, default=200,
            help="number of training epochs")
    args = parser.parse_args()
    print(args)

    main(args)