import argparse
import time
import numpy as np
import torch
import snoop
import pynvml
import sys
from model import PyGT_TGCN
from seastar.dataset.LinkPredDataLoader import LinkPredDataLoader
from seastar.benchmark_tools.table import BenchmarkTable
from utils import to_default_device, get_default_device

def main(args):

    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")
        quit()
    
    if args.dataset == "math":
        dataloader = LinkPredDataLoader('dynamic-temporal', 'sx-mathoverflow-data', args.cutoff_time, verbose=True)
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
    table = BenchmarkTable(f"(PyGT Dynamic-Temporal) TGCN on {dataloader.name} dataset", ["Epoch", "Time(s)", "MSE", "Used GPU Memory (Max MB)", "Used GPU Memory (Avg MB)"])

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

            table.add_row([epoch, "{:.5f}".format(run_time_this_epoch), "{:.4f}".format(sum(cost_arr)/len(cost_arr)), "{:.4f}".format((max(gpu_mem_arr) * 1.0 / (1024**2))),  "{:.4f}".format(((sum(gpu_mem_arr) * 1.0) / ((1024**2) * len(gpu_mem_arr))))])

        table.display()
        print('Average Time taken: {:6f}'.format(np.mean(dur)))

    except RuntimeError as e:
        if 'out of memory' in str(e):
            table.add_row(["OOM", "OOM", "OOM", "OOM",  "OOM"])
            table.display()
        else:
            print("😔 Something went wrong")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seastar Static TGCN')
    snoop.install(enabled=False)

    parser.add_argument("--dataset", type=str, default="math",
            help="Name of the Dataset (math)")
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
    main(args)