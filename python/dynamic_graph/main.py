from timeit import default_timer as timer

# from rich.pretty import pprint
from dynamic_graph import DynamicGraph


graph = DynamicGraph(
    graph_type="pcsr",
    graph_file_path="./PCSR/datasets/test_graph.txt"
)

print("\nIntial Graph\n")
graph.print_graph()
graph.print_array()

start = timer()
graph_dict = graph.get_csr_arrays()
end = timer()
pprint(graph_dict)

print(f'\nTime taken to create CSR Arrays: {end-start}\n')