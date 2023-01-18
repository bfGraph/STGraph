import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl import transform
from dgl.data import register_data_args, load_data

#from gcn import GCN
#from gcn_mp import GCN
from gcn_spmv import GCN, EglGCN

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
    '''
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))
    '''

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

    '''
    # graph preprocess and calculate normalization factor
    g = data.graph
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    '''
    u = edges[:,0]
    v = edges[:,1]
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(u, v)
    # add self loop
    if args.self_loop:
        g = transform.add_self_loop(g)

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    model = GCN(g,
                num_feats,
                args.num_hidden,
                n_classes,
                args.num_layers,
                F.relu,
                args.dropout)
    
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
        model.train()
        torch.cuda.synchronize()
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
        run_time_this_epoch = time.time() - t0
        
        if epoch >= 3:
            dur.append(run_time_this_epoch)	

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        print('Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb'.format(
            epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
        ))


    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.mean(dur)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    #add_argument --dataset
    register_data_args(parser)

    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
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
