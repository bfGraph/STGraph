"""
    Dataset Builder for Soorah - Random Sparse Graphs
"""


"""

{
    'edge_mapping' : {
        'edge_index': {
            't1': [[src, dst] ... ],
            't2': [[src, dst] ... ],
            .
            .
            .
            'tT': [[src, dst] ... ]
        },
        'edge_weight': {
            't1': [w1, w2 ... ],
            't2': [w1, w2 ... ],
            .
            .
            .
            'tT': [w1, w2 ...],
        }
    }
    'time_periods': T,
    'y': [
        [y1, y2 ... ],
        [y1, y2 ... ],
        .
        .
        .
        [y1, y2 ... ]
    ]
}

"""

import json
from random import randint, shuffle

from rich import inspect

eng_covid_file = open("eng_covid.json")
eng_covid = json.load(eng_covid_file)


def create_graph(num_nodes, edge_multiplier, add_coeff, del_coeff, total_time):
    graph_json = {
        "edge_mapping": {"edge_index": {}, "edge_weight": {}},
        "time_periods": total_time,
        "y": [],
    }

    max_edges = (num_nodes * (num_nodes - 1)) * edge_multiplier
    num_edges_low_limit = 0.9 * max_edges
    num_edges_upp_limit = 1.1 * max_edges

    # creating the base graph
    edge_list = []
    edge_weight_list = []
    feature_list = []

    # this will have the number of edges in the graph
    # for the current time stamp
    curr_time_edge_count = randint(num_edges_low_limit, num_edges_upp_limit)

    while len(edge_list) != curr_time_edge_count:
        edge = [randint(0, num_nodes - 1), randint(0, num_nodes - 1)]

        if edge not in edge_list:
            edge_list.append(edge)
            edge_weight_list.append(randint(1, 1000))

    # getting the node features for each node of the base graph
    feature_list = [randint(0, 100) for i in range(num_nodes)]

    graph_json["edge_mapping"]["edge_index"]["0"] = edge_list
    graph_json["edge_mapping"]["edge_weight"]["0"] = edge_weight_list
    graph_json["y"].append(feature_list)

    # creating the next time stamp info from previous
    # time stamps information
    prev_edge_list = edge_list
    for time in range(1, total_time):
        shuffle(prev_edge_list)
    
        del_edge_count = int(len(prev_edge_list) * del_coeff)
        add_edge_count = int(len(prev_edge_list) * add_coeff)
        
        curr_edge_list = prev_edge_list[del_edge_count:]
        curr_edge_weight_list = [randint(1, 1000) for i in range(len(curr_edge_list))]
        curr_time_edge_count = len(curr_edge_list) + add_edge_count
        
        while len(curr_edge_list) != curr_time_edge_count:
            edge = [randint(0, num_nodes - 1), randint(0, num_nodes - 1)]
            
            if edge not in curr_edge_list:
                curr_edge_list.append(edge)
                curr_edge_weight_list.append(randint(1, 1000))
                
        curr_feature_list = [randint(0, 100) for i in range(num_nodes)]
        
        graph_json["edge_mapping"]["edge_index"][str(time)] = curr_edge_list
        graph_json["edge_mapping"]["edge_weight"][str(time)] = curr_edge_weight_list
        graph_json["y"].append(curr_feature_list)
        
        prev_edge_list = curr_edge_list
        
    inspect(graph_json)

create_graph(100, 0.2, 0.1, 0.05, 10)