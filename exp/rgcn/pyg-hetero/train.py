import argparse
import os.path as osp

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv, FastRGCNConv




class Net(torch.nn.Module):
    def __init__(self, RGCNConv, graph, num_layers, num_nodes, num_classes, num_relations, num_hidden):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        #input layer
        self.layers.append(RGCNConv(num_nodes, num_hidden, num_relations,
                              num_bases=None))
        for idx in range(self.num_layers - 2):
            self.layers.append(RGCNConv(num_hidden, num_hidden, num_relations,
                               num_bases=None))
        #outpu layer
        self.output_layer = RGCNConv(num_hidden, num_classes, num_relations,
                              num_bases=None)
        

    def forward(self, edge_index, edge_type):
        x = None
        for idx in range(self.num_layers - 1):
            x = F.relu(self.layers[idx](x, edge_index, edge_type))
        x = self.output_layer(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)





def main():
    #load dataset 
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
    parser.add_argument("--n_layers", type=int, default=2,
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

    args = parser.parse_args()
    #fp = parser.add_mutually_exclusive_group(required=False)
    #fp.add_argument('--validation', dest='validation', action='store_true')
    #fp.add_argument('--testing', dest='validation', action='store_false')

    # Trade memory consumption for faster computation.
    Conv = RGCNConv
    path = './dataset/{}'.format(args.dataset)
    with open('{}/num.txt'.format(path),'r') as f:
        data_list = f.read().split('#')

    num_nodes = int(data_list[0])
    num_relations = int(data_list[1])
    num_classes = int(data_list[2])
    
    labels = np.load('{}/labels.npy'.format(path))#data.labels
    labels = labels[:,0]

    labels = torch.tensor(labels, dtype=torch.long)
    train_idx = np.load('{}/trainIdx.npy'.format(path))#data.train_idx
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    #test_idx = np.load('{}/testIdx.npy'.format(path))#data.test_idx
    # edge type and normfalization factor

    edge_type = np.load('{}/edgeType.npy'.format(path))
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    #edge_norm = np.load('{}/edgeNorm.npy'.format(path))

    edge_src = np.load('{}/edgeSrc.npy'.format(path))
    edge_dst = np.load('{}/edgeDst.npy'.format(path))
    edge_index = torch.tensor(np.vstack([edge_src, edge_dst]), dtype=torch.long)
    #build model 
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') if args.dataset == 'AM' else device
    '''
    #RGCNConv, graph, num_nodes, num_classes, num_relations, num_hidden
    model = Net(Conv, edge_index, args.n_layers, num_nodes, num_classes, num_relations, args.hidden_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    #copy model and data to GPU
    if args.gpu < 0:
        pass
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model.cuda()
        edge_type = edge_type.cuda()
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        edge_type = edge_type.cuda()
        edge_index = edge_index.cuda()
        
    #train code
    dur = []
    for epoch in range(args.num_epochs):
    
        model.train()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()

        out = model(edge_index, edge_type)
        loss = F.nll_loss(out[train_idx], labels[train_idx])
        loss.backward()

        torch.cuda.synchronize()
        train_time_epoch = time.time() - t1
        dur.append(train_time_epoch)
        optimizer.step()
        
        print('Epoch {:05d} Time {:5f}'.format(epoch, train_time_epoch))
    
    Used_memory = torch.cuda.max_memory_allocated(args.gpu) / (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.mean(dur[5:])))


if __name__ == '__main__':
    main()
