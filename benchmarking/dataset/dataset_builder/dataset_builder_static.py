"""
    Dataset Builder for Soorah - Random Sparse Graphs
"""

import json
import os
from random import randint, shuffle, getrandbits
from argparse import ArgumentParser
import time
from tqdm import tqdm

from rich import inspect

from rich.console import Console

console = Console()

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
    '-T',
    help="Total time stamps to be in the dynamic graph dataset",
    default=100,
    metavar="total_timestamps",
    type=int
)

args = parser.parse_args()

def create_graph(
    num_nodes=500,
    edge_multiplier=0.2,
    total_time=100,
):
    graph_json = {
        "edges": [],
        "weights": [],
        "time_periods": total_time,
    }

    edge_count = (num_nodes * (num_nodes - 1)) * edge_multiplier

    # creating the base graph
    edge_list = set()
    edge_weight_list = []
    feature_list = []

    # this will have the number of edges in the graph
    # for the current time stamp
    curr_time_edge_count = int(edge_count)

    console.log("Going to create the edge list")

    while len(edge_list) != curr_time_edge_count:
        edge = (randint(0, num_nodes - 1), randint(0, num_nodes - 1))
        edge_list.add(edge)

    console.log("Finished creating the edge list")

    # adding self loops
    for i in range(num_nodes):
        edge_list.add((i,i))

    console.log("Added self-loops")

    edge_weight_list = [randint(1, 1000) for i in range(len(edge_list))]
    
    console.log("Generated edge weights")
    
    edge_list = list(edge_list)
    
    graph_json["edges"] = edge_list
    graph_json["weights"] = edge_weight_list
    
    for time_stamp in tqdm(range(total_time)):
        feature_list = [randint(0, 100) for i in range(num_nodes)]
        graph_json[str(time_stamp)] = {"y": feature_list}

    console.log("Generated features")
    # inspect(graph_json)

    return graph_json

graph_name = args.dataset_name
dataset_path = "../dataset/"

if os.path.exists(dataset_path + graph_name):
    print("Dataset already exists")
    quit()


t0 = time.time()

graph_json = create_graph(args.N, args.M, args.T)

folder_path = os.path.join(dataset_path, graph_name)
os.mkdir(folder_path)

with open(f"{folder_path}/{graph_name}.json", "w") as fp:
    json.dump(graph_json, fp)

t1 = time.time()

print(f'Total time taken: {t1-t0}')