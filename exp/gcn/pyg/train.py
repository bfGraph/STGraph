import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import numpy as np
import time

'''
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)
'''

class Net(torch.nn.Module):
    def __init__(self, args, num_features, num_classes):
        super(Net, self).__init__()
        self.num_layers = args.num_layers
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, args.num_hidden, 
                               cached=False, normalize=True))
        
        for  i in range(1, self.num_layers):
            self.gcn_layers.append(GCNConv(args.num_hidden, args.num_hidden, 
                               cached=False, normalize=True))
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.output_layer = GCNConv(args.num_hidden, num_classes,
                                    cached=False, normalize=True)

        #self.reg_params = self.conv1.parameters()
        #self.non_reg_params = self.conv2.parameters()

    def forward(self, inputs, edge_index, edge_attr=None):
        #edge_attr is None
        
        x, edge_index, edge_weight = inputs, edge_index, edge_attr
        for i in range(self.num_layers):
            x = F.relu(self.gcn_layers[i](x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
        #x = F.relu(self.conv1(x, edge_index, edge_weight))
        #x = F.dropout(x, training=self.training)

        x = self.output_layer(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

if __name__ == '__main__':
    #argument

    parser = argparse.ArgumentParser(description='GCN')

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

    #load and process data
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

    #print('cuda is available', torch)
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
    #(self, args, num_features, num_classes)
    model = Net(args, num_features, num_classes)

    if args.gpu < 0:
        pass
    else :
        model = model.cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay)

    train_time = 0
    record_time = 0
    Used_memory = 0
    #train

    for epoch in range(args.num_epochs):

        model.train()
        torch.cuda.synchronize()
        tf1 = time.time()

        logits = model(features, edges)
        optimizer.zero_grad()
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