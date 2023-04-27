# Toy Dynamic Dataset

import numpy as np
from rich.pretty import pprint
from rich import inspect

edges_list = []
edges_list.append(
    np.array([
        [0, 1, 2, 4],
        [1, 3, 4, 5]
    ])
)
edges_list.append(
    np.array([
        [2, 1, 3, 2, 3],
        [0, 3, 2, 4, 5]
    ])
)
edges_list.append(
    np.array([
        [0, 1, 0, 3, 3],
        [3, 3, 2, 4, 5]
    ])
)
edges_list.append(
    np.array([
        [0, 2, 5],
        [4, 3, 1]
    ])
)

# ---------------------------------

edges_weight_list = []
edges_weight_list.append(
    np.array(
        [1, 2, 3, 4]
    )
)
edges_weight_list.append(
    np.array(
        [1, 2, 3, 4, 5]
    )
)
edges_weight_list.append(
    np.array(
        [1, 2, 3, 4, 5]
    )
)
edges_weight_list.append(
    np.array(
        [1, 2, 3]
    )
)

# ---------------------------------

all_features_list = []
all_features_list.append(
    np.array(
        [[4], [3], [5], [7], [3], [2]]
    )
)
all_features_list.append(
    np.array(
        [[3], [2], [7], [4], [3], [2]]
    )
)
all_features_list.append(
    np.array(
        [[1], [3], [1], [7], [2], [3]]
    )
)
all_features_list.append(
    np.array(
        [[2], [3], [1], [5], [6], [7]]
    )
)

# ---------------------------------

all_targets_list = []
all_targets_list.append(
    np.array(
        [[2], [1], [6], [2], [4], [3]]
    )
)
all_targets_list.append(
    np.array(
        [[2], [6], [8], [10], [4], [4]]
    )
)
all_targets_list.append(
    np.array(
        [[2], [4], [6], [8], [5], [8]]
    )
)
all_targets_list.append(
    np.array(
        [[5], [8], [9], [6], [9], [10]]
    )
)

# ---------------------------------

# inspect(edges_list)
# inspect(edges_weight_list)
# inspect(all_features_list)
# inspect(all_targets_list)

def get_doorah_dataset():
    return edges_list, edges_weight_list, all_features_list, all_targets_list

def preprocess_graph_structure(edges):
    # inspect(edges)
    tmp_set = set()
    for i in range(len(edges)):
        tmp_set = set()
        for j in range(len(edges[i][0])):
            tmp_set.add(edges[i][0][j])
            tmp_set.add(edges[i][1][j])
    max_num_nodes = len(tmp_set)

    edge_dict = {}
    for i in range(len(edges)):
        edge_set = set()
        for j in range(len(edges[i][0])):
            edge_set.add((edges[i][0][j],edges[i][1][j]))
        edge_dict[str(i)] = edge_set
    
    edge_final_dict = {}
    edge_final_dict["0"] = {"add": list(edge_dict["0"]),"delete": []}
    for i in range(1,len(edges)):
        edge_final_dict[str(i)] = {"add": list(edge_dict[str(i)].difference(edge_dict[str(i-1)])), "delete": list(edge_dict[str(i-1)].difference(edge_dict[str(i)]))}
    
    return edge_final_dict, max_num_nodes

operations, max_num_nodes = preprocess_graph_structure(edges_list)
# inspect(operations)
# inspect(max_num_nodes)