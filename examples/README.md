# Your first STGraph program

NOTE: This tutorial is still in progress

In this beginner friendly tutorial, you will be writing your first GNN and TGNN models and training them on real life graph datasets. Make sure to have `stgraph` installed before continuing with the tutorial.

## Writing a GCN Model

Open up your favourite text editor or Python IDE and create a file named `model.py` with the following code which defines a GCN layer with PyTorch as the backend.

**model.py**
```python
import torch.nn as nn
from stgraph.nn.pytorch.graph_conv import GraphConv


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation):
        super(GCN, self).__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation))

        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation))
        
        self.layers.append(GraphConv(n_hidden, n_classes, None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
```

**utils.py**

```python
import torch


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def to_default_device(data):
    if isinstance(data, (list, tuple)):
        return [to_default_device(x, get_default_device()) for x in data]

    return data.to(get_default_device(), non_blocking=True)
```

The following `train.py` utilizes the GCN layer we defined earlier using STGraph to train on the CORA dataset for a node level classification task

**train.py**
```python
import argparse, time
import numpy as np
import torch
import pynvml
import torch.nn as nn
import torch.nn.functional as F
from stgraph.graph.static.StaticGraph import StaticGraph
from stgraph.dataset.CoraDataLoader import CoraDataLoader
from utils import to_default_device, accuracy
from model import GCN


def main(args):
    cora = CoraDataLoader(verbose=True)

    # To account for the initial CUDA Context object for pynvml
    tmp = StaticGraph([(0, 0)], [1], 1)

    features = torch.FloatTensor(cora.get_all_features())
    labels = torch.LongTensor(cora.get_all_targets())

    train_mask = cora.get_train_mask()
    test_mask = cora.get_test_mask()

    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()

    print("Features Shape: ", features.shape, flush=True)
    edge_weight = [1 for _ in range(len(cora.get_edges()))]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    g = StaticGraph(cora.get_edges(), edge_weight, features.shape[0])
    graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

    # A simple sanity check
    print("Measuerd Graph Size (pynvml): ", graph_mem, " B", flush=True)
    print(
        "Measuerd Graph Size (pynvml): ", (graph_mem) / (1024**2), " MB", flush=True
    )

    # normalization
    degs = torch.from_numpy(g.weighted_in_degrees()).type(torch.int32)
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = to_default_device(norm)
    g.set_ndata("norm", norm.unsqueeze(1))

    num_feats = features.shape[1]
    n_classes = int(max(labels) - min(labels) + 1)
    print("Num Classes: ", n_classes)

    model = GCN(g, num_feats, args.num_hidden, n_classes, args.num_layers, F.relu)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    dur = []
    Used_memory = 0

    for epoch in range(args.num_epochs):
        torch.cuda.reset_peak_memory_stats(0)
        model.train()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        # forward
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        now_mem = torch.cuda.max_memory_allocated(0) + graph_mem
        Used_memory = max(now_mem, Used_memory)

        if cuda:
            torch.cuda.synchronize()

        run_time_this_epoch = time.time() - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        print(
            "Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb ".format(
                epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
            )
        )

    Used_memory /= 1024**3
    print("^^^{:6f}^^^{:6f}".format(Used_memory, np.mean(dur)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")

    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--dataset", type=str, help="Datset to train your model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--num_hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--self-loop", action="store_true", help="graph self-loop (default=False)"
    )
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)

```