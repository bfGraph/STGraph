from PCSR import pcsr

# from rich.pretty import pprint
# from rich import inspect

def pretty_print_pcsr(graph):
    print("Edges:")
    inspect(graph.edges)

    print("Items:")
    for edge in graph.edges.items:
        inspect(edge)

    print("Nodes:")
    inspect(graph.nodes)

    for node in graph.nodes:
        inspect(node)

class DynamicGraph:
    def __init__(self, graph_type: str, graph_file_path: str):
        self.graph_type = graph_type
        if self.graph_type == "pcsr":
            self.graph = pcsr.PCSR()
            self.graph.init_graph(graph_file_path)

    def update_graph(self, operations_list):
        for operation in operations_list:
            action, src, dst, value = operation

            # add an edge
            if action == "ae":
                if self.graph_type == "pcsr":
                    self.graph.add_edge(src, dst, value)

    def print_graph(self):
        if self.graph_type == "pcsr":
            self.graph.print_graph()

    def print_array(self):
        if self.graph_type == "pcsr":
            self.graph.print_array()

    def get_csr_arrays(self):
        if self.graph_type == "pcsr":
            return self.graph.get_csr_arrays()

    def debug_graph(self):
        if self.graph_type == "pcsr":
            pretty_print_pcsr(self.graph)