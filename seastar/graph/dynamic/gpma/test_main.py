from seastar.graph.dynamic.gpma.gpma import GPMA, init_gpma, load_graph, print_gpma_info, edge_update_list, label_edges, copy_label_edges, get_csr_ptrs
from rich import inspect

# initialsing the GPMA graph 
# g = GPMA()
# init_gpma(g, 10)   # num of nodes

# edge_update_list(g, [
#     (0,1),
#     (0,2),
#     (0,3),
#     (1,2),
#     (2,4),
#     (3,6),
#     (6,9),
#     (6,8),
#     (7,9)
# ], is_delete=False, is_reverse_edge=True)

# label_edges(g)

# for i in range(10):
#     print_gpma_info(g, i)

#########################################

# initialsing the GPMA graph 
g = GPMA()
init_gpma(g, 10)   # num of nodes

g1 = GPMA()
init_gpma(g1, 10)   # num of nodes

edge_update_list(g, [
    (1,2),
    (3,2),
    (4,3),
    (4,5),
    (5,1)
])

label_edges(g)

edge_update_list(g1, [
    (2,1),
    (2,3),
    (3,4),
    (5,4),
    (1,5)
])

copy_label_edges(g1,g)

# for i in range(6):
#     print_gpma_info(g, i)

print("###################")

# for i in range(6):
#     print_gpma_info(g1, i)


print(g1.row_offset_ptr)
print("\nHOLAAA")