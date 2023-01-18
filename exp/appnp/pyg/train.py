import argparse
import torch
import time
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP

from train_eval import run, random_planetoid_splits
from datasets import get_planetoid_dataset

'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()
'''
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

class Net(torch.nn.Module):
    def __init__(self, dropout, k, alpha, hiddens, num_features, num_classes):
        super(Net, self).__init__()
        self.dropout = dropout
        self.k = k
        self.alpha = alpha
        self.hiddens = hiddens
        self.appnp_layers = torch.nn.ModuleList()

        self.appnp_layers.append(Linear(num_features, hiddens[0]))

        for i in range(1, len(hiddens)):
            self.appnp_layers.append(Linear(hiddens[i-1], hiddens[i]))

        self.appnp_layers.append(Linear(hiddens[-1], num_classes))
        #self.lin1 = Linear(dataset.num_features, args.hidden)
        #self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(k, alpha)

    def reset_parameters(self):
        for layer in self.appnp_layers:
            layer.reset_parameters()

    def forward(self, inputs, edge_index):
        torch.cuda.synchronize()
        x, edge_index = inputs, edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.appnp_layers[0](x))
        for layer in self.appnp_layers[1:-1]:
            x = F.relu(layer(x))
        x= self.appnp_layers[-1](F.dropout(x, p=self.dropout, training=self.training))
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    #add argement
    parser = argparse.ArgumentParser(description='APPNP')

    parser.add_argument('--dataset', type=str, required=False,
                        help="The input dataset, can be cora, citeseer, pubmed, reddit and so on")
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
    #self, dropout, k, alpha, hiddens, num_features, num_classes
    model = Net(args.in_drop, args.k, args.alpha, args.hidden_sizes, num_features, num_classes)

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