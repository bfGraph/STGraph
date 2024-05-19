import argparse
import time
import numpy as np
import pandas as pd
import torch
import snoop
import pynvml
import sys
import os
import traceback

from .model import STGraphTGCN
from stgraph.graph.static.static_graph import StaticGraph

from stgraph.dataset import WindmillOutputDataLoader
from stgraph.dataset import WikiMathDataLoader
from stgraph.dataset import HungaryCPDataLoader
from stgraph.dataset import PedalMeDataLoader
from stgraph.dataset import METRLADataLoader
from stgraph.dataset import MontevideoBusDataLoader

from stgraph.benchmark_tools.table import BenchmarkTable
from .utils import to_default_device, get_default_device

from rich import inspect

# torch.autograd.set_detect_anomaly(True)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(
    dataset: str,
    num_hidden: int,
    feat_size: int,
    lr: float,
    backprop_every: int,
    num_epochs: int,
    output_file_path: str,
) -> int:
    with open(output_file_path, "w") as f:
        if torch.cuda.is_available():
            print("🎉 CUDA is available", file=f)
        else:
            print("😔 CUDA is not available", file=f)
            return 1

        Graph = StaticGraph([(0, 0)], [1], 1)

        if dataset == "WikiMath":
            dataloader = WikiMathDataLoader(cutoff_time=100)
        elif dataset == "WindMill_large":
            dataloader = WindmillOutputDataLoader(cutoff_time=100)
        elif dataset == "Hungary_Chickenpox":
            dataloader = HungaryCPDataLoader()
        elif dataset == "PedalMe":
            dataloader = PedalMeDataLoader()
        elif dataset == "METRLA":
            dataloader = METRLADataLoader()
        elif dataset == "Montevideo_Bus":
            dataloader = MontevideoBusDataLoader()
        else:
            print("😔 Unrecognized dataset", file=f)
            return 1

        edge_list = dataloader.get_edges()
        edge_weight_list = dataloader.get_edge_weights()
        targets = dataloader.get_all_targets()
        assert not np.isnan(targets).any(), "Targets contain NaN values"

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        G = StaticGraph(edge_list, edge_weight_list, dataloader.gdata["num_nodes"])
        graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

        edge_weight = to_default_device(
            torch.unsqueeze(torch.FloatTensor(edge_weight_list), 1)
        )
        edge_weight = torch.clamp(edge_weight, min=1e-6) # Clamp edge weights
        targets = to_default_device(torch.FloatTensor(np.array(targets)))

        num_hidden_units = num_hidden
        num_outputs = 1
        model = to_default_device(STGraphTGCN(feat_size, num_hidden_units, num_outputs))
        model.apply(init_weights) # Apply custom weight initialization

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Logging Output
        total_timestamps = dataloader.gdata["total_timestamps"]

        if backprop_every == 0:
            backprop_every = total_timestamps - dataloader._lags

        if total_timestamps % backprop_every == 0:
            num_iter = int(total_timestamps / backprop_every)
        else:
            num_iter = int(total_timestamps / backprop_every) + 1

        # metrics
        dur = []
        max_gpu = []
        table = BenchmarkTable(
            f"(STGraph Static-Temporal) TGCN on {dataloader.name} dataset",
            ["Epoch", "Time(s)", "MSE", "Used GPU Memory (Max MB)"],
        )

        # normalization
        degs = torch.from_numpy(G.in_degrees()).type(torch.int32)
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = to_default_device(norm)
        G.set_ndata("norm", norm.unsqueeze(1))

        # nan_table_before = BenchmarkTable("NaN and Inf Truth Table Before Training", ["(index, k)", "y_hat", "edge_weight", "hidden_state"])
        # nan_table_after = BenchmarkTable("NaN and Inf Truth Table After Training", ["(index, k)", "y_out", "y_hat", "hidden_state"])

        # train
        print("Training...\n", file=f)
        try:
            for epoch in range(num_epochs):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                model.train()

                t0 = time.time()
                gpu_mem_arr = []
                cost_arr = []

                for index in range(num_iter):
                    optimizer.zero_grad()
                    cost = 0
                    hidden_state = None
                    y_hat = torch.randn(
                        (dataloader.gdata["num_nodes"], feat_size),
                        device=get_default_device(),
                    )
                    for k in range(backprop_every):
                        t = index * backprop_every + k

                        if t >= total_timestamps - dataloader._lags:
                            break

                        if dataset == "METRLA" and t >= total_timestamps - (dataloader._num_timesteps_out + dataloader._num_timesteps_in):
                            break

                        # # Checking before training for nan and inf
                        # y_hat_has_nan = torch.isnan(y_hat).any() if y_hat is not None else False
                        # y_hat_has_inf = torch.isinf(y_hat).any() if y_hat is not None else False
                        # edge_weight_has_nan = torch.isnan(edge_weight).any() if edge_weight is not None else False
                        # edge_weight_has_inf = torch.isinf(edge_weight).any() if edge_weight is not None else False
                        # hidden_state_has_nan = torch.isnan(hidden_state).any() if hidden_state is not None else False
                        # hidden_state_has_inf = torch.isinf(hidden_state).any() if hidden_state is not None else False
                        #
                        # y_hat_result = "NaN" if y_hat_has_nan else ("Inf" if y_hat_has_inf else "-")
                        # edge_weight_result = "NaN" if edge_weight_has_nan else ("Inf" if edge_weight_has_inf else "-")
                        # hidden_state_result = "NaN" if hidden_state_has_nan else ("Inf" if hidden_state_has_inf else "-")
                        #
                        # nan_table_before.add_row([f"({index}, {k})",y_hat_result, edge_weight_result, hidden_state_result])

                        y_out, y_hat, hidden_state = model(
                            G, y_hat, edge_weight, hidden_state
                        )

                        # # Checking for nans in forward pass
                        # for name, param in model.named_parameters():
                        #     if torch.isnan(param).any():
                        #         print(f"❌ NaNs in {name}")

                        # # Checking after training for nan and inf
                        # y_hat_has_nan = torch.isnan(
                        #     y_hat).any() if y_hat is not None else False
                        # y_hat_has_inf = torch.isinf(
                        #     y_hat).any() if y_hat is not None else False
                        # hidden_state_has_nan = torch.isnan(
                        #     hidden_state).any() if hidden_state is not None else False
                        # hidden_state_has_inf = torch.isinf(
                        #     hidden_state).any() if hidden_state is not None else False
                        # y_out_has_nan = torch.isnan(y_out).any() if y_out is not None else False
                        # y_out_has_inf = torch.isinf(y_out).any() if y_out is not None else False
                        #
                        # y_hat_result = "NaN" if y_hat_has_nan else (
                        #     "Inf" if y_hat_has_inf else "-")
                        # hidden_state_result = "NaN" if hidden_state_has_nan else (
                        #     "Inf" if hidden_state_has_inf else "-")
                        # y_out_result = "NaN" if y_out_has_nan else (
                        #     "Inf" if y_out_has_inf else "-")
                        #
                        # nan_table_after.add_row([f"({index}, {k})", y_out_result, y_hat_result, hidden_state_result])

                        cost = cost + torch.mean((y_out - targets[t]) ** 2)
                        # print(f"Epoch: {epoch}, index: {index}, k: {k}, cost: {cost}", file=f)

                    if cost == 0:
                        break

                    cost = cost / (backprop_every + 1)

                    cost.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Add gradient clipping

                    # # debugging the values of the parameters
                    # print(f"Epoch: {epoch} | Index: {index}\n", file=f)
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Gradient of {name}:\n{param.grad}\n",
                    #               file=f)
                    #     else:
                    #         print(f"No gradient for {name}", file=f)
                    #     if param.requires_grad:
                    #         print(f"Parameter: {name}\n{param.data}\n", file=f)
                    # print(
                    #     "-----------------------------------------------------\n\n",
                    #     file=f)

                    optimizer.step()
                    torch.cuda.synchronize()
                    cost_arr.append(cost.item())

                    # if np.isnan(cost_arr).any():
                    #     breakpoint()

                used_gpu_mem = torch.cuda.max_memory_allocated(0) + graph_mem
                gpu_mem_arr.append(used_gpu_mem)

                run_time_this_epoch = time.time() - t0

                if epoch >= 3:
                    dur.append(run_time_this_epoch)
                    max_gpu.append(max(gpu_mem_arr))

                table.add_row(
                    [
                        epoch,
                        "{:.5f}".format(run_time_this_epoch),
                        "{:.4f}".format(sum(cost_arr) / len(cost_arr)),
                        "{:.4f}".format((max(gpu_mem_arr) * 1.0 / (1024**2))),
                    ]
                )

            table.display(output_file=f)
            print("Average Time taken: {:6f}".format(np.mean(dur)), file=f)

            # nan_table_before.display(output_file=f)
            # nan_table_after.display(output_file=f)

            return 0

        except Exception as e:
            print("---------------- Error ----------------\n", file=f)
            print(e, file=f)
            print("\n", file=f)

            traceback.print_exc(file=f)
            print("\n", file=f)

            if "out of memory" in str(e):
                table.add_row(["OOM", "OOM", "OOM", "OOM"])
                table.display(output_file=f)

            return 1
