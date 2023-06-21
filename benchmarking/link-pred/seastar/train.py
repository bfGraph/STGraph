import json
import urllib.request
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
from seastar.dataset.LinkPredDatasetLoader import LinkPredDatasetLoader
from seastar.nn.pytorch.temporal.tgcn import TGCN
from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.naive.NaiveGraph import NaiveGraph
import pynvml

class SeastarTGCN(torch.nn.Module):
    def __init__(self, node_features, multiplier):
        super(SeastarTGCN, self).__init__()
        self.temporal = TGCN(node_features, multiplier*node_features)
        self.linear = torch.nn.Linear(multiplier*node_features, node_features)

    def forward(self, g, node_feat, edge_weight, hidden_state):
        h = self.temporal(g, node_feat, edge_weight, hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h
  
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def to_default_device(data):
    if isinstance(data,(list,tuple)):
        return [to_default_device(x,get_default_device()) for x in data]
    
    # removed non_blocking
    # (ORIGINAL) data.to(get_default_device(),non_blocking = True)
    return data.to(get_default_device())

def main(args):

    dataloader = LinkPredDatasetLoader(args.dataset_dir, args.dataset, args.num_nodes, verbose=True, for_seastar=True)
    print("Loaded dataset into the train.py seastar", flush=True)

    # to allocate initial CUDA context object
    if args.type == "naive":
        Graph = NaiveGraph([[(0,0)]],1)
    elif args.type == "pcsr":
        Graph = PCSRGraph([[(0,0)]],1)
    elif args.type == "gpma":
        Graph = GPMAGraph([[(0,0)]],1)
    

    train_edges_lst = dataloader.get_edges()
    pos_neg_edges_lst, pos_neg_targets_lst = dataloader.get_pos_neg_edges()
    train_features = to_default_device(torch.randn((dataloader._max_num_nodes, args.feat_size)))
    edge_weight = None
    print("Features ready", flush=True)
    
    model = to_default_device(SeastarTGCN(args.feat_size, args.multiplier))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model ready", flush=True)

    # metrics
    dur = []
    cuda = True
    backprop_every = args.backprop_every

    pos_neg_edges_lst = [to_default_device(torch.from_numpy(pos_neg_edges)) for pos_neg_edges in pos_neg_edges_lst]
    pos_neg_targets_lst = [to_default_device(torch.from_numpy(pos_neg_targets).type(torch.float32)) for pos_neg_targets in pos_neg_targets_lst]
    print("Edge lists ready", flush=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = 0
    graph_mem = 0

    if args.measure_space:
        initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used

    if args.type == "naive":
        G = NaiveGraph(train_edges_lst, args.num_nodes)
    elif args.type == "pcsr":
        G = PCSRGraph(train_edges_lst, args.num_nodes)
    elif args.type == "gpma":
        G = GPMAGraph(train_edges_lst, args.num_nodes)
    else:
        print("Error: Invalid Type")
        quit()

    if args.measure_space:
        graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
        print("Measuerd Graph Size (pynvml): ", graph_mem, " B")
        print("Measuerd Graph Size (pynvml): ", (graph_mem)/(1024**2), " MB")

    # train
    print("Training...\n")
    for epoch in range(args.num_epochs):
        torch.cuda.reset_peak_memory_stats(0)
        model.train()

        if cuda:
            torch.cuda.synchronize()
        
        t0 = time.time()
        cost = 0
        hidden_state = None
        optimizer.zero_grad()
        gpu_mem_arr = []
        y_hat = train_features
        graph_update_time = 0
        forward_prop_time = 0

        # num_iter = int(len(pos_neg_targets) / backprop_every)
        num_iter = len(pos_neg_targets_lst)
        for index in range(num_iter): 
            if args.measure_space:
                initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used

            tGstart = time.time()
            G.get_graph(index)
            graph_update_time += time.time() - tGstart

            if args.measure_space:
                graph_mem_delta = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
                graph_mem = graph_mem + graph_mem_delta

            if G.get_ndata("norm") is None:
                degs = torch.from_numpy(G.in_degrees()).type(torch.float32)
                norm = torch.pow(degs, -0.5)
                norm[torch.isinf(norm)] = 0
                norm = to_default_device(norm)
                G.set_ndata("norm", norm.unsqueeze(1))
            
            # for k in range(backprop_every):
            #     timestamp = index*backprop_every + k

                # if timestamp >= dataloader.total_timestamps:
                #     continue
            timestamp = index
            y_hat, hidden_state = model(G, y_hat, edge_weight, hidden_state)
            out = model.decode(y_hat, pos_neg_edges_lst[timestamp]).view(-1)
            cost = cost + criterion(out, pos_neg_targets_lst[timestamp])

                
        forward_prop_time = time.time() - t0
        cost = cost / (num_iter+1)
        cost.backward()
        optimizer.step()

        if cuda:
            torch.cuda.synchronize()
            
            # Detaching output from the graph so that it isnt backpropagated
            # y_hat = y_hat.detach()
            # hidden_state = hidden_state.detach()
        
        if args.measure_space:
            used_gpu_mem = torch.cuda.max_memory_allocated(0) + graph_mem
            gpu_mem_arr.append(used_gpu_mem)

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

        if args.measure_space:
            print('Epoch {:03d} | Time(s) {:.4f} | BCE {:.2f} | Used GPU Memory (Max) {:.3f} mb | Used GPU Memory (Avg) {:.3f} mb | SeastarGraph (GPU Mem) {:.3f} mb'.format(
                epoch, run_time_this_epoch, cost, (max(gpu_mem_arr) * 1.0 / (1024**2)), ((sum(gpu_mem_arr) * 1.0) / ((1024**2) * len(gpu_mem_arr))), ((graph_mem * 1.0)/(1024**2))
            ))
        else:
            print('Epoch {:03d} | Time(s) {:.4f} | Graph Update Time(s) {:.4f} | Forward Prop Time (s) {:.4f} | BCE {:.2f} '.format(epoch, run_time_this_epoch, graph_update_time, forward_prop_time, cost))

    print('Average Time taken: {:6f}'.format(np.mean(dur)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')

    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--type", type=str, default="naive", help="Seastar Type")
    parser.add_argument("--feat_size", type=int, default=8,
            help="feature size")
    parser.add_argument("--multiplier", type=int, default=2,
            help="multiplier")
    parser.add_argument("--num_epochs", type=int, default=1,
            help="number of training epochs")
    parser.add_argument("--num_nodes", type=int, default=0,
            help="number of nodes")
    parser.add_argument("--backprop_every", type=int, default=1,
            help="number of timestamps after which backprop should happen")
    parser.add_argument("--measure_space", action='store_true',
            help="Measure space")
    parser.add_argument(
        "--dataset_dir", type=str, default="wiki-talk", help="dataset directory"
    )
    parser.add_argument("--dataset", type=str, default="wiki-talk-temporal",
            help="Name of the Dataset", metavar="dataset")
    args = parser.parse_args()
    print(args)

    main(args)