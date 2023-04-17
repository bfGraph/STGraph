import argparse
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from dgl.contrib.data import load_data
from functools import partial

class EGLRGCNModel(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_rels, num_edges, num_bases, dropout, activation):
        super(EGLRGCNModel, self).__init__()

        self.layer1 = RelGraphConv(in_dim,
                                 hidden_dim,
                                 num_rels,
                                 num_bases=num_bases,
                                 dropout=dropout,
                                 activation=activation,
                                 self_loop=False)
        self.layer2 = RelGraphConv(in_dim,
                                 out_dim,
                                 num_rels,
                                 num_bases=num_bases,
                                 dropout=dropout,
                                 activation=activation,
                                 self_loop=False)
        
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        
    def forward(self, g, feats, edge_type, edge_norm):
        # x = self.emb(feats)
        # print(x.shape)
        h = self.layer1(g, feats, edge_type, edge_norm)
        h = self.layer2(g, h, edge_type, edge_norm)
        return h

def main(args):
    # load graph data
    data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes).unsqueeze(1).float()

    # edge type and normalization factor
    edge_type = torch.from_numpy(data.edge_type).long()

    edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1).float()
    labels = torch.from_numpy(labels).view(-1).long()

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()

    # create graph
    g = dgl.graph((data.edge_src,data.edge_dst), num_nodes=num_nodes)
    g = g.to(feats.device)

    model = EGLRGCNModel(num_nodes,
                         1,
                        args.hidden_size,
                        num_classes,
                        num_rels,
                        edge_type.size(0),
                        num_bases=args.num_bases,
                        activation=F.relu,
                        dropout=args.dropout)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    train_idx = np.array([0])

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    train_labels=labels[train_idx]
    train_idx = list(train_idx)
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        tb = time.time()
        train_logits=logits[train_idx]
        ta = time.time()
        loss = F.cross_entropy(train_logits, train_labels)
        t1 = time.time()
        loss.backward()
        optimizer.step()
        # torch.cuda.synchronize()
        t2 = time.time()
        if epoch >=3:
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                  format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print('max memory allocated', torch.cuda.max_memory_allocated())

    model.eval()
    logits = model(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    Used_memory = torch.cuda.max_memory_allocated(0)/(1024**3)
    avg_run_time = np.mean(forward_time[len(forward_time) // 4:]) + np.mean(backward_time[len(backward_time) // 4:])
    #output we need
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--hidden_size", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--num_bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--num_epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
