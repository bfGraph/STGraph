import argparse
import time
import numpy as np
import torch
import snoop
import pynvml
import sys
from model import SeastarTGCN
from seastar.graph.static.StaticGraph import StaticGraph
from seastar.dataset.WindmillOutputDataLoader import WindmillOutputDataLoader
from seastar.dataset.WikiMathDataLoader import WikiMathDataLoader
from seastar.benchmark_tools.table import BenchmarkTable
from utils import to_default_device

def main(args):

    if torch.cuda.is_available():
        print("🎉 CUDA is available")
    else:
        print("😔 CUDA is not available")
        quit()

    # Dummy object to account for CUDA context object
    Graph = StaticGraph([(0,0)], [1], 1)
    
    if args.dataset == "wiki":
        dataloader = WikiMathDataLoader('static-temporal', 'wikivital_mathematics', args.feat_size, args.cutoff_time, verbose=True, for_seastar=True)
    elif args.dataset == "windmill":
        dataloader = WindmillOutputDataLoader('static-temporal', 'windmill_output', args.feat_size, args.cutoff_time, verbose=True, for_seastar=True)
    else:
        print("😔 Unrecognized dataset")
        quit()

    edge_list = dataloader.get_edges()
    edge_weight_list = dataloader.get_edge_weights()
    features = dataloader.get_all_features()
    targets = dataloader.get_all_targets()
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    G = StaticGraph(edge_list, edge_weight_list, dataloader.num_nodes)
    graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

    edge_weight = to_default_device(torch.unsqueeze(torch.FloatTensor(edge_weight_list), 1))
    features = to_default_device(torch.FloatTensor(np.array(features)))
    targets = to_default_device(torch.FloatTensor(np.array(targets)))

    num_hidden_units = args.num_hidden
    num_outputs = 1
    model = to_default_device(SeastarTGCN(args.feat_size, num_hidden_units, num_outputs))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging Output
    print("Dataset: ", args.dataset)
    print("Num Nodes: ", dataloader.num_nodes)
    print("Num Edges: ", len(edge_list))
    print("Num Timestamps: ", dataloader.total_timestamps)

    backprop_every = args.backprop_every
    if backprop_every == 0:
        backprop_every = len(features)
    
    if len(features) % backprop_every == 0:
                num_iter = int(len(features)/backprop_every)
    else:
        num_iter = int(len(features)/backprop_every) + 1

    # metrics
    dur = []
    table = BenchmarkTable(f"(Seastar Static-Temporal) TGCN on {dataloader.name} dataset", ["Epoch", "Time(s)", "MSE", "Used GPU Memory (Max MB)", "Used GPU Memory (Avg MB)"])
    
    # normalization
    degs = torch.from_numpy(G.in_degrees()).type(torch.int32)
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = to_default_device(norm)
    G.set_ndata('norm', norm.unsqueeze(1))

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
                for k in range(backprop_every):
                    t = index * backprop_every + k

                    if t >= len(features):
                        break

                    y_hat, hidden_state = model(G, features[t], edge_weight, hidden_state)
                    cost = cost + torch.mean((y_hat-targets[t])**2)
        
                cost = cost / (backprop_every+1)
                cost.backward()
                optimizer.step()
                torch.cuda.synchronize()
                cost_arr.append(cost.item())

            used_gpu_mem = torch.cuda.max_memory_allocated(0) + graph_mem
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

    parser.add_argument("--dataset", type=str, default="wiki",
            help="Name of the Dataset (wiki, windmill)")
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