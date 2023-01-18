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
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl import transform
from egl_gat import EglGAT
from utils import EarlyStopping


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
        return accuracy(logits, labels)


def train(args):
    # load and preprocess dataset
    path = './dataset/' + str(args.dataset) + '/'
    '''
    edges = np.loadtxt(path + 'edges.txt')
    edges = edges.astype(int)

    features = np.loadtxt(path + 'features.txt')

    train_mask = np.loadtxt(path + 'train_mask.txt')
    train_mask = train_mask.astype(int)

    labels = np.loadtxt(path + 'labels.txt')
    labels = labels.astype(int)
    '''
    edges = np.load(path + 'edges.npy')
    features = np.load(path + 'features.npy')
    train_mask = np.load(path + 'train_mask.npy')
    labels = np.load(path + 'labels.npy')

    num_edges = edges.shape[0]
    num_nodes = features.shape[0]
    num_feats = features.shape[1]
    n_classes = int(max(labels) - min(labels) + 1)

    assert train_mask.shape[0] == num_nodes

    print('dataset {}'.format(args.dataset))
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

    u = edges[:,0]
    v = edges[:,1]

    #initialize a DGL graph
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(u, v)
    
    if isinstance(g, nx.classes.digraph.DiGraph):
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
    elif isinstance(g, DGLGraph):
        g = transform.add_self_loop(g)

    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = EglGAT(g,
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
        now_mem = torch.cuda.max_memory_allocated(0)
        print('now_mem : ', now_mem)
        Used_memory = max(now_mem, Used_memory)
        tf1 =time.time()

        optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 =time.time()
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
        '''
        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):   
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc /{:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))
        
        '''
    

    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    #OUTPUT we need
    avg_run_time = avg_run_time *1. / record_time
    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
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
