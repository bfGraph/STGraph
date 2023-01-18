import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

import numpy as np
import time

class Net(torch.nn.Module):
    def __init__(self, args, graph, num_features, num_classes):
        super(Net, self).__init__()
        self.graph = graph
        self.num_layers = args.num_layers
        self.gat_layers = torch.nn.ModuleList()
        self.in_drop = args.in_drop
        self.gat_layers.append(GATConv(num_features, args.num_hidden, 
                                       heads=args.num_heads, dropout=args.attn_drop))
        self.dropout = F.dropout
        self.elu = F.elu
        self.log_softmax = F.log_softmax
        for i in range(1, self.num_layers):
            self.gat_layers.append(GATConv(args.num_hidden * args.num_heads,
                                   args.num_hidden, heads=args.num_heads,
                                   dropout=args.attn_drop))

        self.output_layer = GATConv(args.num_hidden * args.num_heads, num_classes, 
                                    heads=args.num_out_heads, concat=False,
                                    dropout=args.attn_drop)

    def forward(self, inputs):

        x = inputs

        x = F.dropout(x, p=self.in_drop, training=self.training)

        for i in range(self.num_layers):

            x = F.elu(self.gat_layers[i](x, self.graph))
            #x = F.dropout(x, p=0.6, training=self.training)
        x = self.output_layer(x, self.graph)

        return F.log_softmax(x, dim=1)





def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')

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
    parser.add_argument('--dataset', type=str, required=False,
                        help="The input dataset, can be cora, citeseer, pubmed, reddit and so on")

    args = parser.parse_args()
    print(args)
    #read data

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
    num_features = features.shape[1]
    num_classes = max(labels) - min(labels) + 1

    assert train_mask.shape[0] == num_nodes

    print('dataset {}'.format(args.dataset))
    print('# of edges : {}'.format(num_edges))
    print('# of nodes : {}'.format(num_nodes))
    print('# of features : {}'.format(num_features))

    edges = edges.transpose()

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    edges = torch.LongTensor(edges)
    
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)
    else:
        train_mask = torch.ByteTensor(train_mask)

    print('cuda is available', torch)
    if args.gpu < 0:
        pass
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        edges = edges.cuda()


    #build model
    #(self, args, graph, num_features, num_classes)
    model = Net(args, edges, num_features, num_classes)
    if args.gpu < 0:
        pass
    else :
        model = model.cuda()


    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_time = 0
    record_time = 0
    Used_memory = 0

    #train
    for epoch in range(args.num_epochs):
        model.train()

        torch.cuda.synchronize()
        tf1 = time.time()

        logits = model(features)

        loss = F.nll_loss(logits[train_mask], labels[train_mask])
        now_mem = torch.cuda.max_memory_allocated(0)
        Used_memory = max(Used_memory, now_mem)

        torch.cuda.synchronize()
        tf2 = time.time()

        optimizer.zero_grad()

        torch.cuda.synchronize()    
        tb1 = time.time()

        loss.backward()

        torch.cuda.synchronize()
        tb2 = time.time()
        time_this_epoch = (tf2 - tf1) + (tb2 - tb1)
        if epoch >= 3:
            train_time += time_this_epoch
            record_time += 1

        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        log = 'Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb'

        print(log.format(epoch, time_this_epoch, train_acc, now_mem / (1024**2)))


    #OUTPUT we need
    avg_run_time = train_time * 1. / record_time
    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))
