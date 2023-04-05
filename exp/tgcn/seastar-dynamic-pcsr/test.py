from seastar_graph.engcovid_log import england_covid_log
from seastar_graph.seastar_graph import SeastarGraph
from rich import inspect

graph = SeastarGraph(england_covid_log, 129)
inspect(graph.row_offset)
inspect(graph.column_indices)

graph.update_graph_forward()
inspect(graph.row_offset)
inspect(graph.column_indices)

# graph.base_graph.print_graph()

graph.init_reverse_graph()
graph.update_graph_backward()
inspect(graph.row_offset)
inspect(graph.column_indices)

# graph.reverse_base_graph.print_graph()