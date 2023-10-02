import argparse
import time
import numpy as np
import pandas as pd
import torch
import snoop
import pynvml
import sys
import os
from model import PyGT_TGCN
from stgraph.dataset.LinkPredDataLoader import LinkPredDataLoader
from stgraph.benchmark_tools.table import BenchmarkTable
from utils import to_default_device, get_default_device

def main(args):

    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")
        quit()
    
    if args.dataset == "math":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'sx-mathoverflow-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "wikitalk":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'wiki-talk-temporal-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "askubuntu":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'sx-askubuntu-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "superuser":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'sx-superuser-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "stackoverflow":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'sx-stackoverflow-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "reddit_title":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'reddit-title-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "reddit_body":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'reddit-body-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "email":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'email-eu-core-temporal-data-{args.slide_size}', args.cutoff_time, verbose=True)
    elif args.dataset == "bitcoin_otc":
        dataloader = LinkPredDataLoader('dynamic-temporal', f'bitcoin-otc-data-{args.slide_size}', args.cutoff_time, verbose=True)
    else:
        print("😔 Unrecognized dataset")
        quit()

    edge_lists = dataloader.get_edges()
    pos_neg_edges_lists, pos_neg_targets_lists = dataloader.get_pos_neg_edges()

    edge_lists = [to_default_device(torch.from_numpy(edge_list_t)) for edge_list_t in edge_lists]
    pos_neg_edges_lists = [to_default_device(torch.from_numpy(pos_neg_edges)) for pos_neg_edges in pos_neg_edges_lists]
    pos_neg_targets_lists = [to_default_device(torch.from_numpy(pos_neg_targets).type(torch.float32)) for pos_neg_targets in pos_neg_targets_lists]

    total_timestamps = dataloader.total_timestamps
    num_hidden_units = args.num_hidden
    model = to_default_device(PyGT_TGCN(args.feat_size, num_hidden_units))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Logging Graph Details
    print("Dataset: ", args.dataset)
    print("Num Timestamps: ", total_timestamps)

    backprop_every = args.backprop_every
    if backprop_every == 0:
        backprop_every = total_timestamps
    
    if total_timestamps % backprop_every == 0:
        num_iter = int(total_timestamps/backprop_every)
    else:
        num_iter = int(total_timestamps/backprop_every) + 1

    # metrics
    dur = []
    max_gpu = []
    table = BenchmarkTable(f"(PyGT Dynamic-Temporal) TGCN on {dataloader.name} dataset", ["Epoch", "Time(s)", "MSE", "Used GPU Memory (Max MB)"])

    try:
        # train
        print("Training...\n")
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
                y_hat = torch.randn((dataloader.max_num_nodes, args.feat_size), device=get_default_device())
                for k in range(backprop_every):
                    t = index * backprop_every + k

                    # Since last timestamp does not have a prediction following it
                    if t >= total_timestamps - 1:
                        break

                    y_hat, hidden_state = model(y_hat, edge_lists[t], None, hidden_state)
                    out = model.decode(y_hat, pos_neg_edges_lists[t]).view(-1)
                    cost = cost + criterion(out, pos_neg_targets_lists[t])

                if cost == 0:
                    break

                cost = cost / (backprop_every+1)
                cost.backward()
                optimizer.step()
                torch.cuda.synchronize()
                cost_arr.append(cost.item())

            used_gpu_mem = torch.cuda.max_memory_allocated(0)
            gpu_mem_arr.append(used_gpu_mem)
            run_time_this_epoch = time.time() - t0

            if epoch >= 3:
                dur.append(run_time_this_epoch)
                max_gpu.append(max(gpu_mem_arr))

            table.add_row([epoch, "{:.5f}".format(run_time_this_epoch), "{:.4f}".format(sum(cost_arr)/len(cost_arr)), "{:.4f}".format((max(gpu_mem_arr) * 1.0 / (1024**2)))])

        table.display()
        print('Average Time taken: {:6f}'.format(np.mean(dur)))
        return np.mean(dur), (max(max_gpu) * 1.0 / (1024**2))

    except RuntimeError as e:
        if 'out of memory' in str(e):
            table.add_row(["OOM", "OOM", "OOM", "OOM"])
            table.display()
        else:
            print("😔 Something went wrong")
        return "OOM", "OOM"

def write_results(args, time_taken, max_gpu):
    cutoff = "whole"
    if args.cutoff_time < sys.maxsize:
        cutoff = str(args.cutoff_time)
    file_name = f"pygt_{args.dataset}_T{cutoff}_S{args.slide_size}_B{args.backprop_every}_H{args.num_hidden}_F{args.feat_size}"
    df_data = pd.DataFrame([{'Filename': file_name, 'Time Taken (s)': time_taken, 'Max GPU Usage (MB)': max_gpu}])
    
    if os.path.exists('../../results/dynamic-temporal.csv'):
        df = pd.read_csv('../../results/dynamic-temporal.csv')
        df = pd.concat([df, df_data])
    else:
        df = df_data
    
    df.to_csv('../../results/dynamic-temporal.csv', sep=',', index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seastar Static TGCN')
    snoop.install(enabled=False)

    parser.add_argument("--dataset", type=str, default="math",
            help="Name of the Dataset (math, wikitalk, askubuntu, superuser, stackoverflow, email, bitcoin_otc, reddit_title, reddit_body)")
    parser.add_argument("--slide-size", type=str, default="1.0",
            help="Slide Size")
    parser.add_argument("--backprop-every", type=int, default=0,
            help="Feature size of nodes")
    parser.add_argument("--feat-size", type=int, default=8,
            help="Feature size of nodes")
    parser.add_argument("--num-hidden", type=int, default=100,
            help="Number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--cutoff-time", type=int, default=sys.maxsize,
        help="learning rate")
    parser.add_argument("--num-epochs", type=int, default=1,
            help="number of training epochs")
    args = parser.parse_args()
    
    print(args)
    time_taken, max_gpu = main(args)
    write_results(args, time_taken, max_gpu)