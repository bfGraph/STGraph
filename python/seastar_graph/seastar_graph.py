from rich import inspect
import numpy as np

from PCSR.pcsr import PCSR

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
        ''' Return array of node in-degrees dtype='float32'
        
            Returns a Numpy array containing the in-degree value
            for each node. The in-degree at an index corresponds to
            the in-degree value of a node with the index as it's 
            node label.

            In-degree is the number of directed edges going into
            the given node. In other words, number of edges where
            a given node is the destination.

            Arguments:  None

            Returns:    A numpy array containing the in-degrees of
                        each node, datatype is float32
        '''
        return np.array([node.in_degree for node in self.base_graph.nodes], dtype='float32')