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
import os

from rich import inspect
from rich.pretty import pprint
# from rich.traceback import install
# install(show_locals=True)

from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from seastar.dataset.EnglandCOVID import EnglandCOVID

from doorah import get_doorah_dataset

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

    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")

    # Data
    # dataset = EnglandCovidDatasetLoader()
    
    eng_covid = EnglandCOVID()
    
    # edges_lst, edge_weights_lst, all_features, all_targets = get_doorah_dataset()

    # edges_lst is a list of all edges of dynamic dataset, edge_weights_lst is the
    # corresponding edge weights.
    
    # edges_lst, edge_weights_lst, all_features, all_targets = dataset.get_dataset()
    # edges_lst, edge_weights_lst = presort_edge_weights(edges_lst, edge_weights_lst)

    # inspect(all_features)
    # read = input()

    # NOTE: Using Doorah Dataset

    # inspect(edges_lst)
    # inspect(edge_weights_lst)
    # inspect(all_features)
    # inspect(all_targets)

    # read = input()

    # all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    # all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))

    # Hyperparameters
    # NOTE: Split put to 50% for Doorah Dataset
    # train_test_split = 0.8
    # train_test_split = 0.5
    
    # train_test_split for graph
    # train_edges_lst = edges_lst[:int(len(edges_lst) * train_test_split)]
    # test_edges_lst = edges_lst[int(len(edges_lst) * train_test_split):]
    # train_edge_weights_lst = edge_weights_lst[:int(len(edge_weights_lst) * train_test_split)]
    # test_edge_weights_lst = edge_weights_lst[int(len(edge_weights_lst) * train_test_split):]
    
    

    # train_test_split for features
    train_features = all_features[:int(len(all_features) * train_test_split)]
    train_targets = all_targets[:int(len(all_targets) * train_test_split)]
    test_features = all_features[int(len(all_features) * train_test_split):]
    test_targets = all_targets[int(len(all_targets) * train_test_split):]

    # model
    # NOTE: Number of node features changed to 1 for Doorah Dataset
    model = to_default_device(SeastarTGCN(8))
    # model = to_default_device(SeastarTGCN(1))


    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # metrics
    dur = []
    Used_memory = 0
    cuda = True

    train_graph_log_dict, train_max_num_nodes = preprocess_graph_structure(train_edges_lst)
    G = GPMAGraph(train_graph_log_dict,train_max_num_nodes)
    
    # G = PCSRGraph(train_graph_log_dict,train_max_num_nodes)

    # inspect(train_graph_log_dict)
    # inspect(train_max_num_nodes)
    # inspect(G)
    # inspect(len(train_features))

    # read = input()

    # train
    print("Training...\n")
    for epoch in range(args.num_epochs):
        model.train()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        cost = 0
        hidden_state = None
        optimizer.zero_grad()

        # print("🔴🔴 Attempting Forward prop at t={}".format(epoch))

        # dyn_graph_index is dynamic graph index
        for index in range(0,len(train_features)): 
            
            # Getting the graph for a particular timestamp
            G.get_graph(index) 

            # normalization
            degs = torch.from_numpy(G.in_degrees()).type(torch.int32)
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = to_default_device(norm)
            G.ndata['norm'] = norm.unsqueeze(1)
            edge_weight = to_default_device(torch.FloatTensor(train_edge_weights_lst[index]))
            edge_weight = torch.unsqueeze(edge_weight,1)

            # print("🔴🔴 Attempting Forward prop")
            # inspect(G)

            # forward propagation
            y_hat, hidden_state = model(G, train_features[index], edge_weight, hidden_state)
            
            # print("Epoch={} t={} isNan={}".format(epoch, index, torch.isnan(y_hat).any()))
            
            # print("✨✨✨")
            # inspect(y_hat)
            # inspect(hidden_state)
            
            cost = cost + torch.mean((y_hat-train_targets[index])**2)
            
    
        cost = cost / (index+1)

        now_mem = torch.cuda.max_memory_allocated(0)
        Used_memory = max(now_mem, Used_memory)

        # print("🔴🔴 Forward prop completed")


        # print("🔵🔵 Attempting Backward Prop")
        
        # print("\n📨 CSR Pointers:\n")
        # print("⏩ Forward:\n")
        # print(f'row_offset: {G.fwd_row_offset_ptr}')
        # print(f'column_indices: {G.fwd_column_indices_ptr}')
        # print(f'eids: {G.fwd_eids_ptr}')
        
        # print("\n\n🔙 Backward:")
        # print(f'row_offset: {G.bwd_row_offset_ptr}')
        # print(f'column_indices: {G.bwd_column_indices_ptr}')
        # print(f'eids: {G.bwd_eids_ptr}')
        
        cost.backward()
        # print("🟡🟡 Optimizing")
        optimizer.step()
        # print("🔵🔵 Completed Backward Prop")

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
    # print("Evaluating...\n")
    # model.eval()
    # cost = 0

    # test_graph_log_dict, test_max_num_nodes = preprocess_graph_structure(test_edges_lst)
    # G = GPMAGraph(test_graph_log_dict,test_max_num_nodes)
    
    # # G = PCSRGraph(test_graph_log_dict,test_max_num_nodes)

    # predictions = []
    # true_y = []
    # hidden_state=None
    # # dyn_graph_index is dynamic graph index
    # for index in range(len(test_features)):
    #     # normalization
    #     # degs = G.in_degrees().float()
    #     degs = torch.from_numpy(G.in_degrees())
    #     norm = torch.pow(degs, -0.5)
    #     norm[torch.isinf(norm)] = 0
    #     norm = to_default_device(norm)
    #     G.ndata['norm'] = norm.unsqueeze(1)
    #     edge_weight = to_default_device(torch.FloatTensor(test_edge_weights_lst[index]))
    #     edge_weight = torch.unsqueeze(edge_weight,1)
        
    #     # forward propagation
    #     y_hat, hidden_state = model(G, test_features[index], edge_weight, hidden_state)
    #     cost = cost + torch.mean((y_hat-test_targets[index])**2)
    #     predictions.append(y_hat)
    #     true_y.append(test_targets[index])

    # cost = cost / (index+1)
    # cost = cost.item()
    # print("MSE: {:.4f}".format(cost))



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