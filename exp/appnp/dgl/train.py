import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl
from appnp import APPNP

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


def main(args):
    # load and preprocess dataset

    #FIRST, CHECK DATASET
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
    n_classes = max(labels) - min(labels) + 1

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


    # graph preprocess and calculate normalization factor

    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # create APPNP model
    model = APPNP(g,
                  num_feats,
                  args.hidden_sizes,
                  n_classes,
                  F.relu,
                  args.in_drop,
                  args.edge_drop,
                  args.alpha,
                  args.k)

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

        torch.cuda.synchronize()
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        now_mem = torch.cuda.max_memory_allocated(0)
        Used_memory = max(now_mem, Used_memory)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        run_time_this_epoch = t2 - t0

        if epoch >= 3:
            dur.append(run_time_this_epoch)
            
        train_acc = accuracy(logits[train_mask], labels[train_mask])

        print('Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb'.format(
            epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
        ))

    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.mean(dur)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APPNP')
    register_data_args(parser)
    parser.add_argument("--in-drop", type=float, default=0.5,
                        help="input feature dropout")
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge propagation dropout")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[64],
                        help="hidden unit sizes for appnp")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Teleport Probability")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
