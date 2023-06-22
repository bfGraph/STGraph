"""
    Dataset Builder for Soorah - Random Sparse Graphs
"""

import json
import os
from random import randint, shuffle, getrandbits
from argparse import ArgumentParser
import time
from tqdm import tqdm

import numpy as np
from rich import inspect

parser = ArgumentParser(
    description="Create custom Super Large Sparse Graph Dynamic Dataset - Soorah"
)

parser.add_argument(
    'dataset_name',
    help="Name of the dynamic graph dataset",
    metavar="dataset_name"
)

parser.add_argument(
    '-N',
    help="Number of nodes to be present in the graph",
    default=500,
    metavar="num_nodes",
    type=int
)

parser.add_argument(
    '-M',
    help="Muliplier that is used to get number of edges from total possible edges",
    default=0.2,
    metavar="edge_multiplier",
    type=float
)

parser.add_argument(
    '-A',
    help="Coefficient to find number of edges to be added to the current graph",
    default=0.1,
    metavar="add_coeff",
    type=float
)

parser.add_argument(
    '-D',
    help="Coefficient to find number of edges to be deleted from the current graph",
    default=0.05,
    metavar="del_coeff",
    type=float
)

parser.add_argument(
    '-T',
    help="Total time stamps to be in the dynamic graph dataset",
    default=100,
    metavar="total_timestamps",
    type=int
)

parser.add_argument(
    '-L',
    help="Low-limit multiplier for number of edges",
    default=0.9,
    metavar="low_limit",
    type=float
)

parser.add_argument(
    '-U',
    help="Upper-limit multiplier for number of edges",
    default=1.1,
    metavar="upp_limit",
    type=float
)

args = parser.parse_args()

def create_graph(
    dataset_path,
    num_nodes=500,
    edge_multiplier=0.2,
    add_coeff=0.1,
    del_coeff=0.05,
    total_time=100,
    low_limit=0.9,
    upp_limit=1.1,
):
    
    np_edge_index = []
    np_edge_weight = []
    np_y = []
    
    edge_count = (num_nodes * (num_nodes - 1)) * edge_multiplier
    num_edges_low_limit = low_limit * edge_count
    num_edges_upp_limit = upp_limit * edge_count
    
    # creating the base graph
    edge_list = set()
    edge_weight_list = []
    feature_list = []
    
    # this will have the number of edges in the graph
    # for the current time stamp
    curr_time_edge_count = randint(int(num_edges_low_limit), int(num_edges_upp_limit))
    
    while len(edge_list) != curr_time_edge_count:
        edge = (randint(0, num_nodes - 1), randint(0, num_nodes - 1))
        edge_list.add(edge)

    edge_weight_list = [randint(1, 1000) for i in range(len(edge_list))]
    feature_list = [randint(0, 100) for i in range(num_nodes)]
    edge_list = list(edge_list)
    
    np_edge_index.append(edge_list)
    np_edge_weight.append(edge_weight_list)
    np_y.append(feature_list)
    
    # creating the next time stamp info from previous
    # time stamps information
    prev_edge_list = edge_list
    for time_stamp in tqdm(range(1, total_time)):
        shuffle(prev_edge_list)

        del_edge_count = int(len(prev_edge_list) * del_coeff)
        add_edge_count = int(len(prev_edge_list) * add_coeff)

        curr_edge_list = set(prev_edge_list[del_edge_count:])
        curr_edge_weight_list = [randint(1, 1000) for i in range(len(curr_edge_list))]
        curr_time_edge_count = len(curr_edge_list) + add_edge_count

        while len(curr_edge_list) != curr_time_edge_count:
            edge = (randint(0, num_nodes - 1), randint(0, num_nodes - 1))
            curr_edge_list.add(edge)

        # adding self loops
        for i in range(num_nodes):
            curr_edge_list.add((i,i))

        curr_edge_weight_list = [randint(1, 1000) for i in range(len(curr_edge_list))]
        curr_feature_list = [randint(0, 100) for i in range(num_nodes)]
        
        curr_edge_list = list(curr_edge_list)
        
        np_edge_index.append(curr_edge_list)
        np_edge_weight.append(curr_edge_weight_list)
        np_y.append(curr_feature_list)
        
        prev_edge_list = curr_edge_list
    
        
    np_edge_index = np.array(np_edge_index)
    np_edge_weight = np.array(np_edge_weight)
    np_y = np.array(np_y)
    
    with open(f'{dataset_path}/edge_index.npy', 'wb') as f:
        np.save(f, np_edge_index)
        
    with open(f'{dataset_path}/edge_weight.npy', 'wb') as f:
        np.save(f, np_edge_weight)
        
    with open(f'{dataset_path}/y.npy', 'wb') as f:
        np.save(f, np_y)
    
graph_name = args.dataset_name
dataset_path = "../dataset/"

if os.path.exists(dataset_path + graph_name):
    print("Dataset already exists")
    quit()


# t0 = time.time()
folder_path = os.path.join(dataset_path, graph_name)
os.mkdir(folder_path)

create_graph(folder_path, args.N, args.M, args.A, args.D, args.T, args.L, args.U)


# with open(f"{folder_path}/{graph_name}.json", "w") as fp:
#     json.dump(graph_json, fp)

# t1 = time.time()

# print(f'Total time taken: {t1-t0}')