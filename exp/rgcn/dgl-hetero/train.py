"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import os
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as Transform
from dgl.data.rdf import AIFB, MUTAG, BGS, AM
import dgl
from model import EntityClassify

import networkx as nx

def build_graph(mg, src, dst, ntid, etid, ntypes, etypes):
        # create homo graph
    print('Creating one whole graph ...')
    g = dgl.graph((src, dst))
    g.ndata[dgl.NTYPE] = Transform.tensor(ntid)
    g.edata[dgl.ETYPE] = Transform.tensor(etid)
    print('Total #nodes:', g.number_of_nodes())
    print('Total #edges:', g.number_of_edges())
        
    '''
    # rename names such as 'type' so that they an be used as keys
    # to nn.ModuleDict
    etypes = [RENAME_DICT.get(ty, ty) for ty in etypes]
    mg_edges = mg.edges(keys=True)
    mg = nx.MultiDiGraph()
    for sty, dty, ety in mg_edges:
        mg.add_edge(sty, dty, key=RENAME_DICT.get(ety, ety))
    '''

    # convert to heterograph
        
    print('Convert to heterograph ...')
    hg = dgl.to_hetero(g,
                        ntypes,
                        etypes,
                        metagraph=mg)
    print('#Node types:', len(hg.ntypes))
    print('#Canonical edge types:', len(hg.etypes))
    print('#Unique edge type names:', len(set(hg.etypes)))
    return hg
    #self.graph = hg

def load_dataset(dataset):

    path = './dataset/{}'.format(dataset)
    with open('{}/num.txt'.format(path),'r') as f:
        data_list = f.read().split('#')

    num_nodes = int(data_list[0])
    num_rels = int(data_list[1])
    num_classes = int(data_list[2])

    path = os.path.join('./dataset', dataset)
    labels = np.load(os.path.join(path, 'labels.npy'))
    labels = labels[:,0]
    category = 1
    cnt = 0
    for idx in range(len(labels)):
        if labels[idx] == category:
            cnt += 1

    labels_test = np.random.randint(low=0, high=4, size=[cnt])
    print(labels)
    '''
    labels_list = []
    for idx in range(len(labels)):
        labels_list.append(labels[idx])
    '''
    edgeSrc = np.load(os.path.join(path, 'edgeSrc.npy'))

    edgeDst = np.load(os.path.join(path, 'edgeDst.npy'))

    edgeType = np.load(os.path.join(path, 'edgeType.npy'))
    '''
    edgeType_list = []
    for idx in range(len(edgeType)):
        edgeType_list.append(edgeType[idx])
    '''
    train_idx = np.load(os.path.join(path, 'trainIdx.npy'))
    test_idx = np.load(os.path.join(path, 'testIdx.npy'))

    edge_len = len(edgeSrc)

    mg = nx.MultiDiGraph()
    for idx in range(edge_len):

        mg.add_edge(str(labels[edgeSrc[idx]]), str(labels[edgeDst[idx]]), key=str(edgeType[idx]))
    
    num_label = max(labels) - min(labels) + 1


    #num_label = list(np.arange(num_label))
    label_list = list()
    for x in range(num_label):
        label_list.append(str(x))
    
    num_edgeType = max(edgeType) - min(edgeType) + 1
    #num_edgeType = (np.arange(num_edgeType))
    
    edgeType_list = list()
    for x in range(num_edgeType):
        edgeType_list.append(str(x))
    hg = build_graph(mg, edgeSrc, edgeDst, labels, edgeType, label_list, edgeType_list)
    hash_dict = dict()
    '''
    dict_idx = 0
    for idx in train_idx:
        if not hash_dict.has_key(idx):
            hash_dict[idx] = dict_idx
            dict_idx += 1
    for idx in test_idx:
        if not hash_dict.has_key(idx):
            hash_dict[idx] = dict_idx
            dict_idx += 1
    time.sleep(10)

    for idx in test_idx:
        print (labels[idx])
    time.sleep(10)
    '''
    train_idx = np.arange(cnt)
    test_idx = np.arange(cnt)
    train_idx = Transform.tensor(train_idx)
    test_idx = Transform.tensor(test_idx)
    labels = Transform.tensor(labels)
    return hg, str(category), num_classes, train_idx, test_idx, labels 

def main(args):
    #labels? category?
    g, category, num_classes, train_idx,test_idx ,labels = load_dataset(args.dataset)
    print('done loading dataset')
    
    for ntype in g.ntypes:
        print(ntype, type(ntype))
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = EntityClassify(g,
                           args.hidden_size,
                           num_classes,
                           num_bases=args.num_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")

    dur = []
    model.train()

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        th.cuda.synchronize()
        if epoch > 3:
            t0 = time.time()
        logits = model()[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])


        loss.backward()
        optimizer.step()
        th.cuda.synchronize()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    #print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward()[category]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    
    #output we need 
    Used_memory = th.cuda.max_memory_allocated(0)/(1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, np.average(dur)))

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
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
