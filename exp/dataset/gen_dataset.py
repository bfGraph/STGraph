"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import os
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, CoraFull, Coauthor, AmazonCoBuy

#return number of class 
def count_n_classes(labels):
    labels = labels.numpy()
    max_label = labels.max()
    min_label = labels.min()

    return (max_label + 1)

def cut_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction):
    
    new_n_nodes = int(n_nodes * fraction)
    #check_type(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction)


    remove_nodes_list = []

    for x in range(new_n_nodes, n_nodes):
        remove_nodes_list.append(x)
    
    if isinstance(graph, nx.classes.digraph.DiGraph):
        print('graph is DiGraph')
        graph.remove_nodes_from(remove_nodes_list)
    elif isinstance(graph, DGLGraph):

        print('g is DGLGraph')
        graph.remove_nodes(remove_nodes_list)


    features = features[:new_n_nodes]
    labels = labels[:new_n_nodes]


    train_mask = train_mask[:new_n_nodes]


    val_mask = val_mask[:new_n_nodes]
    test_mask = test_mask[:new_n_nodes]

    return graph, features, labels, train_mask, val_mask, test_mask



def extract_dataset():
    parser = argparse.ArgumentParser(description='DATA')
    register_data_args(parser)
    args = parser.parse_args()
    dataset_name = ['cora', 'citeseer', 'pubmed', 'reddit', 'CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']

    print("Now PATH IS ", os.getcwd())
    for name in dataset_name:
        '''
        if os.path.exists(name):
            print('Folder exists. Skipping ' + name)
            continue
        '''
        if name in ['cora', 'citeseer', 'pubmed', 'reddit']:

            args.dataset = name
            print('args.dataset = ', args.dataset)
            if not os.path.exists(name):
                os.mkdir(name)
            os.chdir(name)

            print("Now PATH IS ", os.getcwd())
            
            data = load_data(args)
            features = data.features
            labels = data.labels
            graph = data.graph
            edges = graph.edges
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

            n_nodes = features.shape[0]
            n_edges = data.graph.number_of_edges
            
            if args.dataset == 'reddit':
                graph, features, labels, train_mask, val_mask, test_mask = cut_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, 0.85)


            

            #edge_x = np.append(edge_x, edge_y, axis=1)
            

            
            edges_list = np.array([])
            first_element = True
            if name != 'reddit':
                for item in edges:

                    if first_element:
                        edges_list = np.array([[item[0], item[1]]])
                        first_element = False
                    else :
                        
                        edges_list = np.append(edges_list, np.array([[item[0], item[1]]]), axis = 0)

            if name == 'reddit':
                edges = graph.edges()

                edge_x = edges[0].numpy().reshape((-1, 1))
                print(edge_x.shape)
                edge_y = edges[1].numpy().reshape((-1, 1))
                edges_list = np.hstack((edge_x, edge_y))
                print(edges_list.shape, edge_x.shape, edge_y.shape)

            print('features_shape', features.shape)
            print('labels_shape', labels.shape)
            print('edges_shape', edges_list.shape)
            '''
            np.savetxt('edges.txt', edges_list)
            np.savetxt('features.txt', features)
            np.savetxt('labels.txt', labels)

            np.savetxt('train_mask.txt', train_mask)
            np.savetxt('val_mask.txt', val_mask)
            np.savetxt('test_mask.txt', test_mask)
            '''
            
            np.save('edges.npy', edges_list)
            np.save('features.npy', features)
            np.save('labels.npy', labels)
            
            np.save('train_mask.npy', train_mask)
            
            print('Finish writing dataset', name)
            os.chdir('..')
            print('change to ', os.getcwd())

        else:

            if not os.path.exists(name):
                os.mkdir(name)
            os.chdir(name)

            if name == 'CoraFull':
                data = CoraFull()
            elif name == 'Coauthor_cs':
                data = Coauthor('cs')
            elif name == 'Coauthor_physics':
                data = Coauthor('physics')
            elif name == 'AmazonCoBuy_computers':
                data = AmazonCoBuy('computers')
            elif name == 'AmazonCoBuy_photo':
                data = AmazonCoBuy('photo')
            else:
                raise Exception("No such a dataset {}".format(name))
    
            graph = data.data[0]
            features = torch.FloatTensor(graph.ndata['feat']).numpy()
            labels = torch.LongTensor(graph.ndata['label']).numpy()

            print('dataset ', name)

            features_shape = features.shape
            labels_shape = labels.shape

            n_nodes = features_shape[0]
            edges_u, edges_v = graph.all_edges()

            edges_u = edges_u.numpy()
            edges_v = edges_v.numpy()
            
            edges_list = np.array([])
            first_element = True
            for idx in range(len(edges_u)):
                if first_element:
                    edges_list = np.array([[edges_u[idx], edges_v[idx]]])
                    first_element = False
                else:
                    edges_list = np.append(edges_list, np.array([[edges_u[idx], edges_v[idx]]]), axis = 0)

            print('features_shape', features_shape)
            print('labels_shape', labels_shape)
            print('edges_shape', edges_list.shape)

            train_mask = []
            for x in range(500):
                train_mask.append(True)
            for x in range(n_nodes - 500):
                train_mask.append(False)
            train_mask = np.array(train_mask)

            '''
            np.savetxt('edges.txt', edges_list)
            np.savetxt('features.txt', features)
            np.savetxt('labels.txt', labels)
            np.savetxt('train_mask.txt', train_mask)
            '''
            
            np.save('edges.npy', edges_list)
            np.save('features.npy', features)
            np.save('labels.npy', labels)
            np.save('train_mask.npy', train_mask)            

            print('Finish writing dataset', name)
            os.chdir('..')
            print('change to ', os.getcwd())




        
    
if __name__ == '__main__':

    extract_dataset()


