from seastar_graph.engcovid_log import england_covid_log
from seastar_graph.seastar_graph import SeastarGraph
from rich import inspect

graph = SeastarGraph(england_covid_log, 129)
graph.base_graph.print_graph()

inspect(england_covid_log)

