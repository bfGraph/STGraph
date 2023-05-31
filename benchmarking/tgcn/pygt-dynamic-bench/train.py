import argparse, time
import numpy as np
import networkx as nx
from seastar.dataset.wikimaths import WikiMaths
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import urllib
from tqdm import tqdm
from tgcn import PyGT_TGCN
import snoop
import os
import nvidia_smi
import psutil

from rich import inspect
from rich.pretty import pprint

# from rich.traceback import install
# install(show_locals=True)

from seastar.dataset.SoorahBase import SoorahBase
from seastar.dataset.FoorahBase import FoorahBase
from seastar.dataset.EnglandCOVID import EnglandCOVID


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
    
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    
    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = (psutil.virtual_memory()[3])
    
    
    eng_covid = FoorahBase(args.dataset_dir, args.dataset, verbose=True)
    # eng_covid = EnglandCOVID(verbose=True)
    
    print("Loaded dataset into the train.py pygt")
    
    edge_list = eng_covid.get_edges()
    edge_weight_list = eng_covid.get_edge_weights()
    all_features = eng_covid.get_all_features()
    all_targets = eng_covid.get_all_targets()
    
    all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))
    
    # print("📝📝📝 GPU Memory in just storing features/targets is: {}".format(((nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem) * 1.0)/(1024**2)))
    # print("📝📝📝 CPU Memory in just storing features/targets is: {}".format(((psutil.virtual_memory()[3] - initial_used_cpu_mem) * 1.0)/(1024**2)))
    # print("\n")
    # Hyperparameters
    train_test_split = 0.8
    
    # train_test_split for graph (Graph)
    train_edges_lst = edge_list[:int(len(edge_list) * train_test_split)]
    test_edges_lst = edge_list[int(len(edge_list) * train_test_split):]
    
    train_edge_weights_lst = edge_weight_list[:int(len(edge_weight_list) * train_test_split)]
    test_edge_weights_lst = edge_weight_list[int(len(edge_weight_list) * train_test_split):]

    # train_test_split for features
    train_features = all_features[:int(len(all_features) * train_test_split)]
    train_targets = all_targets[:int(len(all_targets) * train_test_split)]
    
    test_features = all_features[int(len(all_features) * train_test_split):]
    test_targets = all_targets[int(len(all_targets) * train_test_split):]

    model = to_default_device(PyGT_TGCN(args.feat_size))
    # print("📝📝📝 GPU Memory after storing model is: {}".format(((nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem) * 1.0)/(1024**2)))
    # print("📝📝📝 CPU Memory after storing model is: {}".format(((psutil.virtual_memory()[3] - initial_used_cpu_mem) * 1.0)/(1024**2)))
    # print("\n")

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # metrics
    dur = []
    cuda = True
    
    edge_weight_lst = [to_default_device(torch.FloatTensor(edge_weight)) for edge_weight in train_edge_weights_lst]
    train_edges_lst = [to_default_device(torch.from_numpy(np.array(edge_index))) for edge_index in train_edges_lst]
    
    # print("📝📝📝 GPU Memory after storing edge list/weights: {}".format(((nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem) * 1.0)/(1024**2)))
    # print("📝📝📝 CPU Memory after storing edge list/weights: {}".format(((psutil.virtual_memory()[3] - initial_used_cpu_mem) * 1.0)/(1024**2)))
    # print("\n")
    
    # train_edges_lst = [to_default_device(torch.from_numpy(np.array(edge_index).T)) for edge_index in train_edges_lst]

    # G = GPMAGraph(train_edges_lst)
    # G = PCSRGraph(train_edges_lst)
    # G = NaiveGraph(train_edges_lst)

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
        
        gpu_mem_arr = []
        cpu_mem_arr = []

        # dyn_graph_index is dynamic graph index
        for index in range(0,len(train_features)): 
            
            # t1 = time.time()
            
            edge_weight = edge_weight_lst[index]
            train_edges = train_edges_lst[index]

            # forward propagation
            y_hat, hidden_state = model(train_features[index], train_edges, edge_weight, hidden_state)
            cost = cost + torch.mean((y_hat-train_targets[index])**2)
            
            used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
            gpu_mem_arr.append(used_gpu_mem)
            used_cpu_mem = (psutil.virtual_memory()[3]) - initial_used_cpu_mem
            cpu_mem_arr.append(used_cpu_mem)
            
            # run_time_this_timestamp = time.time() - t1
            # print(f"⌛⌛⌛ Takes a total of {run_time_this_timestamp}")
            
            # if index == 1:
            #     quit()
            
        # print("📝📝📝 GPU Memory after training is: {}".format(((nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem)*1.0))/(1024**2))
        # print("📝📝📝 GPU Memory after training is: {}".format(((nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem) * 1.0)/(1024**2)))
        # print("📝📝📝 CPU Memory after training is: {}".format(((psutil.virtual_memory()[3] - initial_used_cpu_mem) * 1.0)/(1024**2)))
        # print("\n")
    
        cost = cost / (index+1)
        
        cost.backward()
        optimizer.step()

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

        print('Epoch {:03d} | Time(s) {:.4f} | MSE {:.2f} | Used GPU Memory (Max) {:.3f} mb | Used GPU Memory (Avg) {:.3f} mb | Used CPU Memory (Max) {:.3f} mb | Used CPU Memory (Avg) {:.3f} mb'.format(
            epoch, run_time_this_epoch, cost, (max(gpu_mem_arr) * 1.0 / (1024**2)), ((sum(gpu_mem_arr) * 1.0) / ((1024**2) * len(gpu_mem_arr))), (max(cpu_mem_arr) * 1.0 / (1024**2)), ((sum(cpu_mem_arr) * 1.0) / ((1024**2) * len(cpu_mem_arr)))
        ))


    print('Average Time taken: {:6f}'.format(np.mean(dur)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)


    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--feat_size", type=int, default=8,
            help="feature size")
    parser.add_argument("--num_epochs", type=int, default=1,
            help="number of training epochs")
    parser.add_argument(
        "--dataset_dir", type=str, default="foorah_large", help="dataset directory"
    )
    parser.add_argument("--dataset", type=str, default="soorah_base",
            help="Name of the Soorah Dataset", metavar="dataset")
    args = parser.parse_args()
    print(args)

    main(args)