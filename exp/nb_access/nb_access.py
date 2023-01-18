import collections
import argparse
import torch
from torch import nn
from torch.autograd import grad
import dgl
from dgl import DGLGraph
import dgl.backend as B
from dgl import function as fn
import numpy as np


def sort_nodes_by_indegree(edges, decreasing=True):
    dst_nodes = edges[:, 1]
    def sort_by_degree(nodes):
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        if decreasing:
            count_sort_id = np.argsort(-counts)
        else:
            count_sort_id = np.argsort(counts)
        #print(' '.join([str(c) for c in counts[count_sort_id]]))
        return unique_nodes[count_sort_id]
    return sort_by_degree(dst_nodes)

def create_heavy_hitter(edges, created_heavy_hitter_id=0, ratio=0.5):
    deg_count = collections.defaultdict(int)
    all_node_set = set()
    created_heavy_hitter_set = set()
    created_heavy_hitter_set.add(created_heavy_hitter_id)
    for e in edges:
        if e[1] == created_heavy_hitter_id:
            created_heavy_hitter_set.add(e[0])
        all_node_set.add(e[0])
        all_node_set.add(e[1])
    added_new_edges = []
    count = len(created_heavy_hitter_set)
    for n in all_node_set:
        if count/len(all_node_set) > ratio:
            break
        if not n in created_heavy_hitter_set:
            added_new_edges.append([n, created_heavy_hitter_id])
            count += 1
    print('All degree:', len(all_node_set), ' current degree:', len(created_heavy_hitter_set), 'heavy hitter degree', count)
    return added_new_edges 

def rename_nodes_by_rank(edges, nid2rank):
    new_edges = []
    for e in edges:
        new_edges.append([nid2rank[e[0]], nid2rank[e[1]]])
    return np.array(new_edges)

def bench_graph(edges, num_nodes, args, features, node_map, deg_inc_node_map):
    #initialize a DGL graph
    u = edges[:,0]
    v = edges[:,1]
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(u, v)
    n_edges = g.number_of_edges()
    ret = B.nb_access_bench(g, features, node_map, deg_inc_node_map)
    feat = args.feature_dim
    while feat > 0:
        features = torch.ones([num_nodes, feat]).cuda()
        ret = B.nb_access_bench(g, features, node_map, deg_inc_node_map)
        feat = feat//2

def main(args):
    path = './dataset/' + str(args.dataset) + '/'
    edges = np.load(path + 'edges.npy')
    features = np.load(path + 'features.npy')
    edges = edges.astype(int)
    if args.heavy_hitter_ratio > 0:
        edges = np.append(edges, create_heavy_hitter(edges, created_heavy_hitter_id=features.shape[0]-1,ratio=args.heavy_hitter_ratio), axis=0) 
        print('create heavy hitter with hit ratio no less than ', args.heavy_hitter_ratio)

    sorted_dst_nodes = sort_nodes_by_indegree(edges)
    rank_of_nodes = {sorted_dst_nodes[i] :i for i in range(len(sorted_dst_nodes))}
    sorted_edges = rename_nodes_by_rank(edges, rank_of_nodes)
    deg_inc_dst_nodes = sort_nodes_by_indegree(edges, decreasing=False)


    num_edges = edges.shape[0]
    num_nodes = features.shape[0]
    num_feats = features.shape[1]
    features = torch.FloatTensor(features)
    torch.cuda.set_device(args.gpu)
    features = features.cuda()
    node_map = torch.from_numpy(sorted_dst_nodes).int().cuda()
    deg_inc_node_map = torch.from_numpy(deg_inc_dst_nodes).int().cuda()

    print('dataset {}'.format(args.dataset))
    print('# of edges : {}'.format(num_edges))
    print('# of nodes : {}'.format(num_nodes))
    print('# of features : {}'.format(num_feats))
    print('Unsorted graph benchmark\n--------------------')
    bench_graph(edges, num_nodes, args, features, node_map, deg_inc_node_map)
    print('Sorted graph benchmark\n--------------------')
    bench_graph(sorted_edges, num_nodes, args, features, node_map, deg_inc_node_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fusedGatUnitTest')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use.")
    parser.add_argument("--dataset", type=str, default='',
                        help="name of dataset option cora, citeseer, pubmed and e.t.c.")
    parser.add_argument("--feature_dim", type=int, default=-1,
                        help="nodes' starting feature dimension for mannually generated features.")
    parser.add_argument("--heavy_hitter_ratio", type=float, default=-1,
                        help="Mannually create heavy hitter whose in-degree/num_node no less than heavy_hitter_ratio. default to be -1")
    args = parser.parse_args()
    print(args)
    main(args)
