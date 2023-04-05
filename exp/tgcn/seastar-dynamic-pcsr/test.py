from seastar_graph.engcovid_log import england_covid_log
from seastar_graph.seastar_graph import SeastarGraph
from rich import inspect

graph = SeastarGraph(england_covid_log, 129)
# graph.base_graph.print_graph()
inspect(graph.num_nodes)
inspect(graph.num_edges)


for timestamp in range(5):
    graph.update_graph_forward()
    inspect(graph.num_nodes)
    inspect(graph.num_edges)
    # inspect(graph.column_indices)
    # graph.base_graph.print_graph()