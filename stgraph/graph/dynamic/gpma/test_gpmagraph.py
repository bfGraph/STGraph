from stgraph.graph.dynamic.gpma.GPMAGraph import GPMAGraph
from seastar.graph.dynamic.gpma.gpma import GPMA, init_gpma, print_gpma_info, edge_update_list, label_edges, copy_label_edges, build_reverse_gpma, load_graph
from rich import inspect
# g = GPMA()
# load_graph(g, "facebook.txt")

# print_gpma_info(g, 0)

graph_updates = {
    "0": {
        "add": [(2,1),(2,5),(2,3)],
        "delete": []
    },
    "1": {
        "add": [(1,5),(5,4),(3,4)],
        "delete": [(2,5)]
    },
    "2": {
        "add": [(3,1)],
        "delete": [(2,1),(1,5),(5,4)]
    }
}

g = GPMAGraph(graph_updates=graph_updates,max_num_nodes=6)

for i in range(6):
    print_gpma_info(g.forward_graph,i)

g.get_forward_graph_for_timestamp(1)

print("#####\n")

for i in range(6):
    print_gpma_info(g.forward_graph,i)

g.get_forward_graph_for_timestamp(2)

print("#####\n")

for i in range(6):
    print_gpma_info(g.forward_graph,i)

print("#####\n")

g.get_backward_graph_for_timestamp(2)

for i in range(6):
    print_gpma_info(g.backward_graph,i)

print("#####\n")

g.get_backward_graph_for_timestamp(1)

for i in range(6):
    print_gpma_info(g.backward_graph,i)

print("#####\n")

g.get_backward_graph_for_timestamp(0)

for i in range(6):
    print_gpma_info(g.backward_graph,i)

print("#####\n")

inspect(g)

print(g.row_offset_ptr)
print(g.column_indices_ptr)
print(g.eids_ptr)
print(g.num_nodes)
print(g.num_edges)

