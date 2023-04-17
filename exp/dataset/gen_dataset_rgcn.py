"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import os
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

def main(dataset):

    # load graph data
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    os.chdir(dataset)
    data = load_data(dataset, bfs_level=3, relabel=False)
    
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx
    print(type(num_nodes))
    print(type(num_rels))
    print(type(num_classes))
    with open('num.txt', 'w') as f:
        f.write('{}#{}#{}'.format(num_nodes, num_rels, num_classes))
    np.save('labels.npy', labels)
    print(type(train_idx))
    np.save('trainIdx.npy', train_idx)
    print(type(test_idx))
    np.save('testIdx.npy', test_idx)
    # split dataset into train, validate, test


    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)

    # edge type and normalization factor
    print('edge_src type = ', type(data.edge_src))
    print('shape = ', data.edge_src.shape)
    print(data.edge_src)
    np.save('edgeSrc.npy', data.edge_src)
    np.save('edgeDst.npy', data.edge_dst)

    print('***')
    print('edge_type type =', type(data.edge_type))
    print('edge_type shape =', data.edge_type.shape)
    print(data.edge_type)
    np.save('edgeType.npy', data.edge_type)
    print('***')
    print('edge_norm type =', type(data.edge_norm))
    print('edge_norm shape = ', data.edge_norm.shape)
    print(data.edge_norm)
    np.save('edgeNorm.npy', data.edge_norm)
    print('***')
    print('Finish extracting dataset : {}'.format(dataset))
    os.chdir('..')

if __name__ == '__main__':
    dataset_list = ['aifb', 'mutag', 'bgs']
    for dataset in dataset_list:
        main(dataset)
