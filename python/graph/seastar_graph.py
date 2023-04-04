from rich import inspect

from pcsr import PCSR
from engcovid_log import england_covid_log

class SeastarGraph:
    def __init__(self, graph_updates: dict, max_num_nodes: dict):
        '''
            Parameters:
                graph_updates (dict)    A python dictionary containing the updates made
                                        to a graph dataset over time. Key is the timestamp(str)
                                        and the value is another dictionary. The inside 
                                        dictionary has key values "add" and "delete" which
                                        contains a list of edges(list) that are added or deleted
                                        respectively for a given timestamp
                
                max_num_nodes (int)     Maximum number of nodes present at any given timestamp
                                        for the graph         
        '''

        self.max_num_nodes = max_num_nodes
        self.base_graph = PCSR(max_num_nodes)


        initial_graph_additions = graph_updates["0"]["add"]

        for edge in initial_graph_additions:
            self.base_graph.add_edge(edge[1], edge[0], 1)

    def in_degrees(self):
        ''' Return list of node in-degrees
        
        '''
        return [node.in_degree for node in self.base_graph.nodes]    
graph = SeastarGraph(england_covid_log, 129)
graph.base_graph.print_graph()

inspect(graph.in_degrees())