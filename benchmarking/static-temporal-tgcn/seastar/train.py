import argparse
import time
import numpy as np
import pandas as pd
import torch
import snoop
import pynvml
import sys
import os

from model import STGraphTGCN
from stgraph.graph.static.StaticGraph_old import StaticGraph

from stgraph.dataset.WindmillOutputDataLoader import WindmillOutputDataLoader
from stgraph.dataset.WikiMathDataLoader import WikiMathDataLoader
from stgraph.dataset.HungaryCPDataLoader import HungaryCPDataLoader
from stgraph.dataset.PedalMeDataLoader import PedalMeDataLoader
from stgraph.dataset.METRLADataLoader import METRLADataLoader
from stgraph.dataset.MontevideoBusDataLoader import MontevideoBusDataLoader

from stgraph.benchmark_tools.table import BenchmarkTable
from utils import to_default_device, get_default_device

from rich import inspect


def main(args):
    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")
        quit()

    # Dummy object to account for CUDA context object
    Graph = StaticGraph([(0, 0)], [1], 1)

    if args.dataset == "wiki":
        dataloader = WikiMathDataLoader(
            "static-temporal",
            "wikivital_mathematics",
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    elif args.dataset == "windmill":
        dataloader = WindmillOutputDataLoader(
            "static-temporal",
            "windmill_output",
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    elif args.dataset == "hungarycp":
        dataloader = HungaryCPDataLoader(
            "static-temporal",
            "HungaryCP",
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    elif args.dataset == "pedalme":
        dataloader = PedalMeDataLoader(
            "static-temporal",
            "pedalme",
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    elif args.dataset == "metrla":
        dataloader = METRLADataLoader(
            "static-temporal",
            "METRLA",
            args.feat_size,
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    elif args.dataset == "monte":
        dataloader = MontevideoBusDataLoader(
            "static-temporal",
            "montevideobus",
            args.feat_size,
            args.cutoff_time,
            verbose=True,
            for_stgraph=True,
        )
    else:
        print("😔 Unrecognized dataset")
        quit()

    edge_list = dataloader.get_edges()
    edge_weight_list = dataloader.get_edge_weights()
    targets = dataloader.get_all_targets()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    G = StaticGraph(edge_list, edge_weight_list, dataloader.num_nodes)
    graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

    edge_weight = to_default_device(
        torch.unsqueeze(torch.FloatTensor(edge_weight_list), 1)
    )
    targets = to_default_device(torch.FloatTensor(np.array(targets)))

    num_hidden_units = args.num_hidden
    num_outputs = 1
    model = to_default_device(
        STGraphTGCN(args.feat_size, num_hidden_units, num_outputs)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging Output
    total_timestamps = dataloader.total_timestamps
    print("Dataset: ", args.dataset)
    print("Num Nodes: ", dataloader.num_nodes)
    print("Num Edges: ", len(edge_list))
    print("Num Timestamps: ", dataloader.total_timestamps)

    backprop_every = args.backprop_every
    if backprop_every == 0:
        backprop_every = total_timestamps

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

    # train
    print("Training...\n")
    try:
        for epoch in range(args.num_epochs):
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
                    (dataloader.num_nodes, args.feat_size), device=get_default_device()
                )
                for k in range(backprop_every):
                    t = index * backprop_every + k

                    if t >= total_timestamps:
                        break

                    y_out, y_hat, hidden_state = model(
                        G, y_hat, edge_weight, hidden_state
                    )
                    cost = cost + torch.mean((y_out - targets[t]) ** 2)

                if cost == 0:
                    break

                cost = cost / (backprop_every + 1)
                cost.backward()
                optimizer.step()
                torch.cuda.synchronize()
                cost_arr.append(cost.item())

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

        table.display()
        print("Average Time taken: {:6f}".format(np.mean(dur)))
        return np.mean(dur), (max(max_gpu) * 1.0 / (1024**2))

    except RuntimeError as e:
        if "out of memory" in str(e):
            table.add_row(["OOM", "OOM", "OOM", "OOM"])
            table.display()
        else:
            print("😔 Something went wrong")
        return "OOM", "OOM"


def write_results(args, time_taken, max_gpu):
    cutoff = "whole"
    if args.cutoff_time < sys.maxsize:
        cutoff = str(args.cutoff_time)
    file_name = f"stgraph_{args.dataset}_T{cutoff}_B{args.backprop_every}_H{args.num_hidden}_F{args.feat_size}"
    df_data = pd.DataFrame(
        [
            {
                "Filename": file_name,
                "Time Taken (s)": time_taken,
                "Max GPU Usage (MB)": max_gpu,
            }
        ]
    )

    if os.path.exists("../../results/static-temporal.csv"):
        df = pd.read_csv("../../results/static-temporal.csv")
        df = pd.concat([df, df_data])
    else:
        df = df_data

    df.to_csv(
        "../../results/static-temporal.csv", sep=",", index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGraph Static TGCN")
    snoop.install(enabled=False)

    parser.add_argument(
        "--dataset",
        type=str,
        default="wiki",
        help="Name of the Dataset (wiki, windmill, hungary_cp, pedalme, metrla, monte)",
    )
    parser.add_argument(
        "--backprop-every", type=int, default=0, help="Feature size of nodes"
    )
    parser.add_argument(
        "--feat-size", type=int, default=8, help="Feature size of nodes"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=100, help="Number of hidden units"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--cutoff-time", type=int, default=sys.maxsize, help="learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="number of training epochs"
    )
    args = parser.parse_args()

    print(args)
    time_taken, max_gpu = main(args)
    write_results(args, time_taken, max_gpu)
