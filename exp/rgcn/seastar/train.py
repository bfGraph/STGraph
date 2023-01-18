"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.contrib.data import load_data
import dgl.backend as B
from functools import partial


"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
class EglRelGraphConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 num_edges,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0,
                 layer_type=0):
        super(EglRelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.layer_type = layer_type

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        self.dropout = nn.Dropout(dropout)


    def forward(self, g, x, etypes, norm=None):
        """ Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :mathtorch.bmm(A.unsqueeze(0).expand_as(v), v):`(|E|, 1)`

        Returns
        -------
        torch.Tensor
            New node features.
        """
        #print('aaa',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        #torch.cuda.synchronize()
        #t1 = time.time()
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            #print('weight size:', self.weight.size(), 'w_comp size:', self.w_comp.size())
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
            #print('new weight size:', weight.size())
        else:
            weight = self.weight
        #torch.cuda.synchronize()
        #print('bbb',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        #t2 = time.time()

        if self.layer_type == 0:
            node_repr = B.rgcn_layer0(g, weight, norm)
            #print('output of layer 0', node_repr)
        else:
            node_repr = B.rgcn_layer1(g, x, weight, norm)
        #torch.cuda.synchronize()
        #t3 = time.time()
            #print('output of layer 1', node_repr)
        #print('ccc',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        #torch.cuda.synchronize()
        #t4 = time.time()
        #print('matmul takes:',t2-t1, 's', (t2-t1)/(t4-t1),'%')
        #print('gcn takes:',t3-t2, 's', (t3-t2)/(t4-t1),'%')
        #print('rest takes:',t4-t3, 's', (t4-t3)/(t4-t1),'%')
        return node_repr

class EGLRGCNModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim, out_dim, num_rels, num_edges, num_bases, dropout, activation):
        super(EGLRGCNModel, self).__init__()
        self.layer1 = EglRelGraphConv(num_nodes,
                                 hidden_dim,
                                 num_rels,
                                 num_edges,
                                 num_bases=num_bases,
                                 dropout=dropout,
                                 activation=activation,
                                 layer_type=0)
        self.layer2 = EglRelGraphConv(hidden_dim,
                                 out_dim,
                                 num_rels,
                                 num_edges,
                                 num_bases=num_bases,
                                 dropout=dropout,
                                 activation=activation,
                                 layer_type=1)
        
    def forward(self, g, feats, edge_type, edge_norm):
        h = self.layer1.forward(g, feats, edge_type, edge_norm)
        h = self.layer2.forward(g, h, edge_type, edge_norm)
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
    feats = torch.arange(num_nodes)

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
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges_with_type(data.edge_src, data.edge_dst, data.edge_type)
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
    model = EGLRGCNModel(num_nodes,
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
        torch.cuda.synchronize()
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
    logits = model.forward(g, feats, edge_type, edge_norm)
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
