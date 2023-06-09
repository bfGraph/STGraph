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


def run_seastar(dataset_dir, dataset, feat_size, lr, type, max_num_nodes, num_epochs):
    print_var.is_print_verbose_log = False

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = psutil.virtual_memory()[3]

    eng_covid = FoorahBase(dataset_dir, dataset, verbose=False, for_seastar=True)

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
    gpu_move_time_dur = []
    
    temp_array = []
    prop_time_dur = []
    
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

        G._gpu_move_time = 0
        cost = 0
        hidden_state = None
        optimizer.zero_grad()

        gpu_mem_arr = []
        cpu_mem_arr = []

        update_time_start = G._total_update_time

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

        update_time_end = G._total_update_time

        if epoch >= 3:
            dur.append(run_time_this_epoch)
            gpu_move_time_dur.append(G._gpu_move_time)
            prop_time_dur.append(update_time_end-update_time_start)  
                  
        # temp_array.append(G._total_update_time)

    if G._update_count == 0:
        time_per_update = 0
    else:
        time_per_update = G._total_update_time / G._update_count

    return np.mean(dur), time_per_update, np.mean(gpu_move_time_dur), np.mean(prop_time_dur)


def run_pygt(dataset_dir, dataset, feat_size, lr, num_epochs):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    initial_used_gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    initial_used_cpu_mem = psutil.virtual_memory()[3]

    eng_covid = FoorahBase(dataset_dir, dataset, verbose=False)

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
    pygt_gpu_move_dur = []
    cuda = True

    edge_weight_lst = [
        to_default_device(torch.FloatTensor(edge_weight))
        for edge_weight in train_edge_weights_lst
    ]
    
    # NOTE: Previously done like this
    move_t0 = time.time()
    train_edges_lst = [
        to_default_device(torch.from_numpy(np.array(edge_index)))
        for edge_index in train_edges_lst
    ]
    move_t1 = time.time()

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
        move_time_total = 0

        # dyn_graph_index is dynamic graph index
        for index in range(0, len(train_features)):
            # t1 = time.time()

            edge_weight = edge_weight_lst[index]
            train_edges = train_edges_lst[index]

            # NOTE: New method for PyG-T
            # move_t0 = time.time()
            # train_edges = to_default_device(torch.from_numpy(np.array(train_edges_lst[index])))
            # move_t1 = time.time()
            
            # move_time_total += (move_t1 - move_t0)

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
            # pygt_gpu_move_dur.append(move_time_total)

    time_per_update = 0
    update_time_epoch = 0

    return np.mean(dur), time_per_update, move_t1-move_t0, 0


def main(args):
    console.log("Starting Benchmarking")

    table = Table(title=f"\nBenchmarking T-GCN\n", show_edge=False, style="black bold")

    table.add_column("Dataset Name", justify="right")
    table.add_column("Feat. Size", justify="left")
    table.add_column("Naive", justify="left")
    table.add_column("PCSR", justify="left")
    table.add_column("GPMA", justify="left")
    table.add_column("PyG-T", justify="left")

    update_time_table = Table(
        title=f"\nTime Taken per Update [black bold](µs)[/black bold]\n", show_edge=False, style="black bold"
    )

    update_time_table.add_column("Dataset Name", justify="right")
    update_time_table.add_column("Feat. Size", justify="left")
    update_time_table.add_column("Naive", justify="left")
    update_time_table.add_column("PCSR", justify="left")
    update_time_table.add_column("GPMA", justify="left")
    update_time_table.add_column("PyG-T", justify="left")
    
    gpu_move_time_table = Table(
        title=f"\nGPU Move Time [black bold](per epoch)[/black bold]\n", show_edge=False, style="black bold"
    )

    gpu_move_time_table.add_column("Dataset Name", justify="right")
    gpu_move_time_table.add_column("Feat. Size", justify="left")
    gpu_move_time_table.add_column("Naive", justify="left")
    gpu_move_time_table.add_column("PCSR", justify="left")
    gpu_move_time_table.add_column("GPMA", justify="left")
    gpu_move_time_table.add_column("PyG-T", justify="left")
    
    time_prop_update_table = Table(
        title=f"\nTotal time taken for update [black bold](per epoch)[/black bold]\n", show_edge=False, style="black bold"
    )

    time_prop_update_table.add_column("Dataset Name", justify="right")
    time_prop_update_table.add_column("Feat. Size", justify="left")
    time_prop_update_table.add_column("Naive", justify="left")
    time_prop_update_table.add_column("PCSR", justify="left")
    time_prop_update_table.add_column("GPMA", justify="left")
    time_prop_update_table.add_column("PyG-T", justify="left")

    dataset_dir = args.dataset_dir

    # creating the dataset list
    dataset_name = {}

    dataset_feat_size = 8
    while dataset_feat_size <= args.max_feat_size:
        dataset_name[f"{dataset_dir}_{dataset_feat_size}"] = {
            "feat_size": dataset_feat_size,
            "max_num_nodes": 55000,
        }
        dataset_feat_size += 8

    seastar_graph_types = ["naive", "pcsr", "gpma"]
    # seastar_graph_types = ["pcsr"]

    learning_rate = args.lr
    num_epochs = args.num_epochs

    for dataset, param in dataset_name.items():
        results = {}
        update_time_results = {}
        gpu_move_time_results = {}
        total_update_time_results = {}

        # running seastar t-gcn
        for graph_type in seastar_graph_types:
            console.log(
                f"Running [bold yellow]{graph_type}[/bold yellow] on [bold cyan]{dataset}[/bold cyan]"
            )
            avg_time, update_time, gpu_move_time, total_update_time = run_seastar(
                dataset_dir,
                dataset,
                param["feat_size"],
                learning_rate,
                graph_type,
                param["max_num_nodes"],
                num_epochs,
            )
            results[graph_type] = round(avg_time, 4)
            update_time_results[graph_type] = round(update_time * (10**6), 4)
            gpu_move_time_results[graph_type] = round(gpu_move_time, 4)
            total_update_time_results[graph_type] = round(total_update_time, 4)

        # running pygt t-gcn
        console.log(
            f"Running [bold yellow]PyG-T[/bold yellow] on [bold cyan]{dataset}[/bold cyan]"
        )

        avg_time, update_time, gpu_move_time, total_update_time = run_pygt(
            dataset_dir,
            dataset,
            param["feat_size"],
            learning_rate,
            num_epochs,
        )

        results["pygt"] = round(avg_time, 4)
        update_time_results["pygt"] = update_time
        gpu_move_time_results["pygt"] = round(gpu_move_time, 4)
        total_update_time_results["pygt"] = round(total_update_time, 4)

        # getting the implementation with the fastest time
        fast_impl = min(results, key=results.get)
        results[fast_impl] = f"[bold green]{results[fast_impl]}[/bold green]"
        
        fast_gpu_move_time = min(gpu_move_time_results, key=gpu_move_time_results.get)
        gpu_move_time_results[fast_gpu_move_time] = f"[bold green]{gpu_move_time_results[fast_gpu_move_time]}[/bold green]"

        table.add_row(
            str(dataset),
            str(param["feat_size"]),
            str(results["naive"]),
            str(results["pcsr"]),
            str(results["gpma"]),
            str(results["pygt"]),
        )

        update_time_table.add_row(
            str(dataset),
            str(param["feat_size"]),
            str(update_time_results["naive"]),
            str(update_time_results["pcsr"]),
            str(update_time_results["gpma"]),
            str(update_time_results["pygt"]),
        )
        
        gpu_move_time_table.add_row(
            str(dataset),
            str(param["feat_size"]),
            str(gpu_move_time_results["naive"]),
            str(gpu_move_time_results["pcsr"]),
            str(gpu_move_time_results["gpma"]),
            str(gpu_move_time_results["pygt"]),
        )
        
        time_prop_update_table.add_row(
            str(dataset),
            str(param["feat_size"]),
            str(total_update_time_results["naive"]),
            str(total_update_time_results["pcsr"]),
            str(total_update_time_results["gpma"]),
            str(total_update_time_results["pygt"]),
        )

    console.print(table)
    console.print(update_time_table)
    console.print(gpu_move_time_table)
    console.print(time_prop_update_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--dataset_dir", type=str, default="foorah_large", help="dataset directory"
    )
    parser.add_argument("--max_feat_size", type=int, default=8, help="max_feature_size")
    parser.add_argument("--max_num_nodes", type=int, default=1000, help="max_num_nodes")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="number of training epochs"
    )
    args = parser.parse_args()

    main(args)