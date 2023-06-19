import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from rich import inspect
from rich.pretty import pprint

from seastar.graph.static.StaticGraph import StaticGraph
from seastar.dataset.cora import CoraDataset

import snoop

from seastar.nn.pytorch.graph_conv import GraphConv
import gc
import sys
import pynvml

class EglGCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(EglGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

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


    
    cora = CoraDataset(verbose=True)

    # To account for the initial CUDA Context object for pynvml
    tmp = StaticGraph([(0,0)], [1], 1)
    a = torch.tensor([2]).cuda()
    
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

    print("Features Shape: ", features.shape)
    edge_weight = [1 for _ in range(len(cora.get_edges()))]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    g = StaticGraph(cora.get_edges(), edge_weight, features.shape[0])
    graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

    # A simple sanity check
    print("Measuerd Graph Size (pynvml): ", graph_mem, " B")
    print("Measuerd Graph Size (pynvml): ", (graph_mem)/(1024**2), " MB")
    
    # add self loop
    # if args.self_loop:
    #     g = dgl.add_self_loop(g)
    
    # g = g.to(features.device)

    # normalization
    degs = torch.from_numpy(g.weighted_in_degrees()).type(torch.int32)
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    
    norm = to_default_device(norm)
    g.ndata['norm'] = norm.unsqueeze(1)
    print("Norm Shape: ", g.ndata['norm'].shape)

    num_feats = features.shape[1]
    n_classes = int(max(labels) - min(labels) + 1)
    print("Num Classes: ",n_classes)

    model = EglGCN(g,
                num_feats,
                args.num_hidden,
                n_classes,
                args.num_layers,
                F.relu)
    
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    Used_memory = 0

    for epoch in range(args.num_epochs):
        torch.cuda.reset_peak_memory_stats(0)
        model.train()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        # print(f"\n\n------- BEFORE MODEL E={epoch} ----------")
        # # gc.collect()
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj):
        #         print(type(obj), obj.size(), obj.device, sys.getrefcount(obj))

        # forward
        logits = model(g, features)

        # print(f"------- AFTER MODEL E={epoch} -------\n\n")
        # # gc.collect()
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj):
        #         print(type(obj), obj.size(), obj.device, sys.getrefcount(obj))

        loss = loss_fcn(logits[train_mask], labels[train_mask])
        # now_mem = torch.cuda.max_memory_allocated(0)
        # Used_memory = max(now_mem, Used_memory)
        # now_mem_pynvml = (
        #         pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem
        #     )

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
        print('Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb '.format(
            epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
        ))

    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.mean(dur)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)

    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--dataset", type=str,
            help="Datset to train your model")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--num_hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--num_layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)