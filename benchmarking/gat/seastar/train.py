"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import time
import torch
import torch.nn.functional as F
import pynvml
from stgraph.graph.static.StaticGraph import StaticGraph
from stgraph.dataset.CoraDataLoader import CoraDataLoader
from utils import EarlyStopping, accuracy
import snoop
import numpy as np
from model import GAT


def train(args):
    cora = CoraDataLoader(verbose=True)

    # To account for the initial CUDA Context object for pynvml
    tmp = StaticGraph([(0,0)], [1], 1)
    
    features = torch.FloatTensor(cora.get_all_features())
    labels = torch.LongTensor(cora.get_all_targets())
    train_mask = cora.get_train_mask()
    test_mask = cora.get_test_mask()
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)
    edges = cora.get_edges()
    edge_weight = [1 for _ in range(len(cora.get_edges()))]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    g = StaticGraph(edges, edge_weight, features.shape[0])
    graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

    num_edges = len(edges)
    num_nodes = features.shape[0]
    num_feats = features.shape[1]
    n_classes = int(max(labels) - min(labels) + 1)

    assert train_mask.shape[0] == num_nodes

    print('dataset {}'.format("Cora"))
    print('# of edges : {}'.format(num_edges))
    print('# of nodes : {}'.format(num_nodes))
    print('# of features : {}'.format(num_feats))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)

    else:
        train_mask = torch.ByteTensor(train_mask)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()

    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    record_time = 0
    avg_run_time = 0
    Used_memory = 0

    for epoch in range(args.num_epochs):
        #print('epoch = ', epoch) 
        #print('mem0 = {}'.format(mem0))
        torch.cuda.synchronize()
        tf = time.time()
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        now_mem = torch.cuda.max_memory_allocated(0) + graph_mem
        Used_memory = max(now_mem, Used_memory)

        optimizer.zero_grad()
        torch.cuda.synchronize()
        loss.backward()
        optimizer.step()
        t2 =time.time()
        run_time_this_epoch = t2 - tf

        if epoch >= 3:
            dur.append(time.time() - t0)
            record_time += 1

            avg_run_time += run_time_this_epoch

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        #log for each step
        print('Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb'.format(
            epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
        ))

    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    #OUTPUT we need
    avg_run_time = avg_run_time *1. / record_time
    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')

    # COMMENT IF SNOOP IS TO BE ENABLED
    snoop.install(enabled=False)

    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()

    print(args)
        
    train(args)
