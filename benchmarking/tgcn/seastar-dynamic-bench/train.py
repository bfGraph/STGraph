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

import nvidia_smi
import psutil

from rich import inspect
from rich.pretty import pprint
from rich.console import Console
from rich.table import Table

# from rich.traceback import install
# install(show_locals=True)

from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from seastar.graph.dynamic.naive.NaiveGraph import NaiveGraph
from seastar.dataset.EnglandCOVID import EnglandCOVID
from seastar.dataset.SoorahBase import SoorahBase
from seastar.dataset.FoorahBase import FoorahBase
from seastar.dataset.wikimaths import WikiMaths

import seastar.compiler.debugging.print_variables as print_var


# GPU | CPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def to_default_device(data):
    if isinstance(data, (list, tuple)):
        return [to_default_device(x, get_default_device()) for x in data]

    return data.to(get_default_device(), non_blocking=True)


def main(args):
    print_var.is_print_verbose_log = args.verbose

    # preparing the benchmarking table
    table = Table(title=f"\nBenchmarking T-GCN with {args.type}\n", show_edge=False, style="black bold")

    table.add_column("Epoch", justify="right")
    table.add_column("Time (s)", justify="left")
    table.add_column("MSE", justify="left")
    table.add_column("Max. GPU Memory (mb)", justify="left")
    table.add_column("Avg. GPU Memory (mb)", justify="left")
    table.add_column("Max. CPU Memory (mb)", justify="left")
    table.add_column("Avg. CPU Memory (mb)", justify="left")

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = psutil.virtual_memory()[3]

    eng_covid = FoorahBase(args.dataset_dir, args.dataset, verbose=True, for_seastar=True)

    print("Loaded dataset into the train.py seastar")

    edge_list = eng_covid.get_edges()
    edge_weight_list = eng_covid.get_edge_weights()
    all_features = eng_covid.get_all_features()
    all_targets = eng_covid.get_all_targets()

    all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))

    # Hyperparameters
    train_test_split = 0.8

    # train_test_split for graph
    train_edges_lst = edge_list[: int(len(edge_list) * train_test_split)]
    test_edges_lst = edge_list[int(len(edge_list) * train_test_split) :]

    train_edge_weights_lst = edge_weight_list[
        : int(len(edge_weight_list) * train_test_split)
    ]
    test_edge_weights_lst = edge_weight_list[
        int(len(edge_weight_list) * train_test_split) :
    ]

    # train_test_split for features
    train_features = all_features[: int(len(all_features) * train_test_split)]
    train_targets = all_targets[: int(len(all_targets) * train_test_split)]

    test_features = all_features[int(len(all_features) * train_test_split) :]
    test_targets = all_targets[int(len(all_targets) * train_test_split) :]

    model = to_default_device(SeastarTGCN(args.feat_size))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # metrics
    dur = []
    cuda = True

    if args.type == "naive":
        G = NaiveGraph(train_edges_lst,args.max_num_nodes)
    elif args.type == "pcsr":
        G = PCSRGraph(train_edges_lst,args.max_num_nodes)
    elif args.type == "gpma":
        G = GPMAGraph(train_edges_lst,args.max_num_nodes)
    else:
        print("Error: Invalid Type")
        quit()

    # inspect(G.graph_updates)
    # num_edges = 0
    # for key, val in G.graph_updates.items():
    #     num_edges += len(val["add"])
    #     num_edges -= len(val["delete"])
    #     print("For timestamp={} NUM_ADDITIONS={} NUM_DELETIONS={} | TOTAL_EDGE_COUNT={}".format(key,len(val["add"]),len(val["delete"]),num_edges),flush=True)
    # quit()

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
        for index in range(0, len(train_features)):
            # t1 = time.time()

            # Getting the graph for a particular timestamp
            G.get_graph(index)

            # normalization
            degs = torch.from_numpy(G.in_degrees()).type(torch.int32)
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = to_default_device(norm)
            G.ndata["norm"] = norm.unsqueeze(1)
            edge_weight = to_default_device(
                torch.FloatTensor(train_edge_weights_lst[index])
            )
            edge_weight = torch.unsqueeze(edge_weight, 1)

            # forward propagation
            y_hat, hidden_state = model(
                G, train_features[index], edge_weight, hidden_state
            )

            cost = cost + torch.mean((y_hat - train_targets[index]) ** 2)

            used_gpu_mem = (
                nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
            )
            gpu_mem_arr.append(used_gpu_mem)
            used_cpu_mem = (psutil.virtual_memory()[3]) - initial_used_cpu_mem
            cpu_mem_arr.append(used_cpu_mem)
            # run_time_this_timestamp = time.time() - t1
            # print(f"⌛⌛⌛ Takes a total of {run_time_this_timestamp}")

            # if index == 1:
            #     quit()

        cost = cost / (index + 1)

        # t1 = time.time()
        # print("⚠️⚠️⚠️ Starting Backprop")
        cost.backward()
        # print("⚠️⚠️⚠️ Backprop Completed")
        # print(f"⌛⌛⌛ Time taken for backprop {time.time() - t1}")

        # if epoch == 1:
        #     quit()

        optimizer.step()

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

        table.add_row(
            str(epoch),
            str(round(run_time_this_epoch, 4)),
            str(round(cost.item(), 4)),
            str(round((max(gpu_mem_arr) * 1.0 / (1024**2)), 4)),
            str(round(((sum(gpu_mem_arr) * 1.0) / ((1024**2) * len(gpu_mem_arr))), 4)),
            str(round((max(cpu_mem_arr) * 1.0 / (1024**2)), 4)),
            str(round(((sum(cpu_mem_arr) * 1.0) / ((1024**2) * len(cpu_mem_arr))), 4)),
        )

        # print(
        #     "Epoch {:03d} | Time(s) {:.4f} | MSE {:.2f} | Used GPU Memory (Max) {:.3f} mb | Used GPU Memory (Avg) {:.3f} mb | Used CPU Memory (Max) {:.3f} mb | Used CPU Memory (Avg) {:.3f} mb".format(
        #         epoch,
        #         run_time_this_epoch,
        #         cost,
        #         (max(gpu_mem_arr) * 1.0 / (1024**2)),
        #         ((sum(gpu_mem_arr) * 1.0) / ((1024**2) * len(gpu_mem_arr))),
        #         (max(cpu_mem_arr) * 1.0 / (1024**2)),
        #         ((sum(cpu_mem_arr) * 1.0) / ((1024**2) * len(cpu_mem_arr))),
        #     )
        # )

    console = Console()
    console.print(table)

    print("\nAverage Time taken (s): {:4f}".format(np.mean(dur)))

    # updates_per_sec = round((G._update_count / G._total_update_time), 0)
    # print(f'Updates per second: {updates_per_sec}')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--feat_size", type=int, default=8, help="feature_size")
    parser.add_argument("--max_num_nodes", type=int, default=1000, help="max_num_nodes")
    parser.add_argument("--type", type=str, default="naive", help="Seastar Type")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of training epochs"
    )
    
    parser.add_argument(
        "--dataset_dir", type=str, default="foorah_large", help="dataset directory"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="foorah_large_8",
        help="Name of the Soorah Dataset",
        metavar="dataset",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="If set to true, will print out logs while Seastar compiles your model",
        metavar="verbose",
    )
    args = parser.parse_args()

    main(args)
