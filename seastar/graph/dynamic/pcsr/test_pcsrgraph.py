# from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
from pcsr import PCSR, build_reverse_pcsr
# from seastar.graph.dynamic.pcsr.PCSRGraph import PCSRGraph
import copy

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

# edge_lst = [[(2,1),(2,5),(2,3)],[(1,5),(5,4),(3,4),(2,1),(2,3)],[(3,4),(2,3),(3,1)]]

# g = PCSRGraph(edge_list=edge_lst)

# print("Graph at time=0  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# g.forward_graph.print_graph()

# g._update_graph_forward()
# print("Graph at time=1  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# g.forward_graph.print_graph()

# g.get_forward_graph_for_timestamp(2)
# print("Graph at time=2  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# g.forward_graph.print_graph()

# g.get_backward_graph_for_timestamp(2)
# print("Graph at time=2  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# print("Forward")
# g.forward_graph.print_graph()
# print("Backward")
# g.backward_graph.print_graph()

# g.get_backward_graph_for_timestamp(1)
# print("Graph at time=1  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# print("Forward")
# g.forward_graph.print_graph()
# print("Backward")
# g.backward_graph.print_graph()

# g.get_backward_graph_for_timestamp(0)
# print("Graph at time=0  num_nodes={}    num_edges={}".format(g.num_nodes,g.num_edges))
# print("Forward")
# g.forward_graph.print_graph()
# print("Backward")
# g.backward_graph.print_graph()

# g = PCSR(6)

# g.edge_update_list(graph_updates["0"]["add"])
# print(f"After first add, num_edges={g.edge_count}")

# g.print_array()

# g1 = copy.deepcopy(g)

# g.edge_update_list(graph_updates["1"]["add"])
# print(f"After second add, num_edges={g.edge_count}")

# g.edge_update_list(graph_updates["1"]["delete"],is_delete=True)
# print(f"After second delete, num_edges={g.edge_count}")

# g.edge_update_list(graph_updates["2"]["delete"],is_delete=True)
# print(f"After third delete, num_edges={g.edge_count}")

# g.edge_update_list(graph_updates["2"]["add"])
# print(f"After third add, num_edges={g.edge_count}")

# g1.print_array()

edge_lst = [[(2,1),(2,5),(2,3)],[(1,5),(5,4),(3,4),(2,1),(2,3)],[(3,4),(2,3),(3,1)]]
print("HEREE",flush=True)
# g = PCSRGraph(edge_list=edge_lst)
# g._update_graph_forward()
# g.current_timestamp += 1
# g._update_graph_forward()
# g.current_timestamp += 1
# g._init_reverse_graph()
# g._is_backprop_state = True
# g._update_graph_backward()
# g.current_timestamp -= 1
# g._update_graph_backward()
# g.current_timestamp -= 1
# print(g._backward_graph.edge_count)

# g.get_graph(0)
# g.get_graph(1)
# g.get_graph(2)
# g.get_backward_graph(2)
# g.get_backward_graph(1)
# g.get_backward_graph(0)
# print(g._forward_graph.edge_count)

fwd = PCSR(6)
bwd = PCSR(6)

fwd.edge_update_list(edge_lst[0], is_reverse_edge=True)
fwd.edge_update_list([(1,5),(5,4),(3,4)], is_reverse_edge=True)
build_reverse_pcsr(bwd,fwd)
print(bwd.edge_count)


