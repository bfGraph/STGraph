import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from seastar_tgcn import SeastarTGCN
from pygt_tgcn import PyGT_TGCN
import snoop
import os

import nvidia_smi
import psutil

from rich import inspect
from rich.pretty import pprint
from rich.console import Console
from rich.table import Table

console = Console()

from seastar.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from seastar.graph.dynamic.naive.NaiveGraph import NaiveGraph
from seastar.dataset.FoorahBase import FoorahBase

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


def run_seastar(dataset, feat_size, lr, type, max_num_nodes, num_epochs):
    print_var.is_print_verbose_log = False

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = psutil.virtual_memory()[3]

    eng_covid = FoorahBase(dataset, verbose=False, for_seastar=True)

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

    model = to_default_device(SeastarTGCN(feat_size))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # metrics
    dur = []
    cuda = True

    if type == "naive":
        G = NaiveGraph(train_edges_lst, max_num_nodes)
    elif type == "pcsr":
        G = PCSRGraph(train_edges_lst, max_num_nodes)
    elif type == "gpma":
        G = GPMAGraph(train_edges_lst, max_num_nodes)
    else:
        print("Error: Invalid Type")
        quit()

    # train
    for epoch in range(num_epochs):
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

        cost = cost / (index + 1)

        cost.backward()
        optimizer.step()

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

    return np.mean(dur)


def run_pygt(dataset, feat_size, lr, num_epochs):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = psutil.virtual_memory()[3]

    eng_covid = FoorahBase(dataset, verbose=False)

    edge_list = eng_covid.get_edges()
    edge_weight_list = eng_covid.get_edge_weights()
    all_features = eng_covid.get_all_features()
    all_targets = eng_covid.get_all_targets()

    all_features = to_default_device(torch.FloatTensor(np.array(all_features)))
    all_targets = to_default_device(torch.FloatTensor(np.array(all_targets)))

    # Hyperparameters
    train_test_split = 0.8

    # train_test_split for graph (Graph)
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

    model = to_default_device(PyGT_TGCN(feat_size))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # metrics
    dur = []
    cuda = True

    edge_weight_lst = [
        to_default_device(torch.FloatTensor(edge_weight))
        for edge_weight in train_edge_weights_lst
    ]
    train_edges_lst = [
        to_default_device(torch.from_numpy(np.array(edge_index)))
        for edge_index in train_edges_lst
    ]

    # train
    for epoch in range(num_epochs):
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

            edge_weight = edge_weight_lst[index]
            train_edges = train_edges_lst[index]

            # forward propagation
            y_hat, hidden_state = model(
                train_features[index], train_edges, edge_weight, hidden_state
            )
            cost = cost + torch.mean((y_hat - train_targets[index]) ** 2)

            used_gpu_mem = (
                nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
            )
            gpu_mem_arr.append(used_gpu_mem)
            used_cpu_mem = (psutil.virtual_memory()[3]) - initial_used_cpu_mem
            cpu_mem_arr.append(used_cpu_mem)

        cost = cost / (index + 1)

        cost.backward()
        optimizer.step()

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

    return np.mean(dur)


def main(args):
    console.log("Starting Benchmarking")

    table = Table(title=f"\nBenchmarking T-GCN\n", show_edge=False, style="black bold")

    table.add_column("Dataset Name", justify="right")
    table.add_column("Feat. Size", justify="left")
    table.add_column("Naive", justify="left")
    table.add_column("PCSR", justify="left")
    table.add_column("PyG-T", justify="left")

    # creating the dataset list
    dataset_name = {}

    dataset_feat_size = 8
    while dataset_feat_size <= args.max_feat_size:
        dataset_name[f"foorah_large_feat_{dataset_feat_size}"] = {
            "feat_size": dataset_feat_size,
            "max_num_nodes": 55000,
        }
        dataset_feat_size += 8

    seastar_graph_types = ["naive", "pcsr"]

    learning_rate = 0.01
    num_epochs = 5

    for dataset, param in dataset_name.items():
        results = {}

        # running seastar t-gcn
        for graph_type in seastar_graph_types:
            console.log(
                f"Running [bold yellow]{graph_type}[/bold yellow] on [bold cyan]{dataset}[/bold cyan]"
            )
            avg_time = run_seastar(
                dataset,
                param["feat_size"],
                learning_rate,
                graph_type,
                param["max_num_nodes"],
                num_epochs,
            )
            results[graph_type] = round(avg_time, 4)

        # running pygt t-gcn
        console.log(
            f"Running [bold yellow]PyG-T[/bold yellow] on [bold cyan]{dataset}[/bold cyan]"
        )

        results["pygt"] = round(
            run_pygt(
                dataset,
                param["feat_size"],
                learning_rate,
                num_epochs,
            ),
            4,
        )

        # getting the implementation with the fastest time
        fast_impl = min(results, key=results.get)
        results[fast_impl] = f"[bold green]{results[fast_impl]}[/bold green]"

        table.add_row(
            str(dataset),
            str(param["feat_size"]),
            str(results["naive"]),
            str(results["pcsr"]),
            str(results["pygt"]),
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--max_feat_size", type=int, default=8, help="max_feature_size")
    parser.add_argument("--max_num_nodes", type=int, default=1000, help="max_num_nodes")
    parser.add_argument("--type", type=str, default="naive", help="Seastar Type")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of training epochs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="soorah_base",
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
