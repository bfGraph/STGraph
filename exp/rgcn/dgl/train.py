"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import time
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
from dgl.contrib.data import load_data
from functools import partial

from model import BaseRGCN

class EntityClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=None,
                self_loop=self.use_self_loop)


def check_type(name, arr):

    print(name, type(arr))

def main(args):

    # load graph data
    #data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    path = './dataset/{}'.format(args.dataset)
    with open('{}/num.txt'.format(path),'r') as f:
        data_list = f.read().split('#')

    num_nodes = int(data_list[0])
    num_rels = int(data_list[1])
    num_classes = int(data_list[2])
    
    labels = np.load('{}/labels.npy'.format(path))#data.labels
    
    train_idx = np.load('{}/trainIdx.npy'.format(path))#data.train_idx
    test_idx = np.load('{}/testIdx.npy'.format(path))#data.test_idx

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)

    # edge type and normfalization factor

    edge_type = np.load('{}/edgeType.npy'.format(path))
    edge_norm = np.load('{}/edgeNorm.npy'.format(path))
    edge_src = np.load('{}/edgeSrc.npy'.format(path))
    edge_dst = np.load('{}/edgeDst.npy'.format(path))

    edge_type = torch.from_numpy(edge_type).long()
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1).long()
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
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(edge_src, edge_dst)
    #tu_forward = sorted(list(zip(data.edge_src, data.edge_dst, data.edge_type)), key=lambda x : (x[1], x[2]))
    #tu_backward = sorted(list(zip(data.edge_dst, data.edge_src,  data.edge_type)), key=lambda x : (x[1], x[2]))
    #def compute_e_to_distict_t(tu):
    #    num_edges = len(tu)
    #    all_node_distinct_types = 0
    #    cur_node = tu[0][1]
    #    type_set = set()
    #    type_set.add(tu[0][2])
    #    for i in range(1, len(tu)):
    #        if tu[i][1] == cur_node:
    #            type_set.add(tu[i][2])
    #        else:
    #            all_node_distinct_types += len(type_set)
    #            cur_node = tu[i][1]
    #            type_set.clear()
    #            type_set.add(tu[i][2])
    #    all_node_distinct_types += len(type_set)
    #    type_set.clear()
    #    #print('\n'.join([str(t) for t in tu]))
    #    print('num_edges:', num_edges, 'node distinct types', all_node_distinct_types)
    #    return num_edges/all_node_distinct_types
    #r_forward = compute_e_to_distict_t(tu_forward)
    #r_backward = compute_e_to_distict_t(tu_backward)
    #print('ratio forward:', r_forward, 'ratio_backward:', r_backward)

    # create model
    model = EntityClassify(len(g),
                           args.hidden_size,
                           num_classes,
                           num_rels,
                           num_bases=args.num_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        print('logits:', logits.size())
        print('labels:', labels.size())
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
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
    logits = model.forward(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    

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
