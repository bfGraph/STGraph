from rich import inspect
import numpy as np

from pcsr.pcsr import PCSR

class SeastarGraph:
    def __init__(self, graph_updates: dict, max_num_nodes: dict):
        ''' Creates the initial base graph

            Creates the initial base graph based on the first time stamp
            provided in the "graph_updates" dictionary. The dynamic graph
            is for now stored in PCSR format. The CSR arrays are then
            generated for the base graph i.e row_offset, column_indices and eids.

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

        self.graph_updates = graph_updates
        self.max_num_nodes = max_num_nodes
        self.base_graph = PCSR(max_num_nodes)
        self.ndata = {}
        self.current_time_stamp = 0

        self.num_nodes = 0
        self.num_edges = 0

        self.row_offset = []
        self.column_indices = []
        self.eids = []

        initial_graph_additions = graph_updates["0"]["add"]

        for edge in initial_graph_additions:
            self.base_graph.add_edge(edge[1], edge[0], 1)

        self._get_graph_csr()
        self._get_num_nodes_edges()

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
    
    def _get_graph_csr(self):
        ''' Generates the CSR Arrays for the base graph
        
            This generates the row_offset, column_indices and eids
            arrays for the current base graph.

            Warning:    As of now, it generates an empty list for eids.
                        Need to work on that in the future.
        '''
        csr_map = self.base_graph.get_csr_arrays()
        self.row_offset = csr_map['row_offset']
        self.column_indices = csr_map['column_indices']

        # TODO: Make changes to work with edge ids
        # self.eids = np.arange(len())

    def update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        
        '''
        self.current_time_stamp += 1

        if self.graph_updates[str(self.current_time_stamp)] == None:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        graph_additions = self.graph_updates[str(self.current_time_stamp)]["add"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp)]["delete"]

        for edge in graph_additions:
            self.base_graph.add_edge(edge[1], edge[0], 1)

        for edge in graph_deletions:
            self.base_graph.delete_edge(edge[1], edge[0])

        self._get_graph_csr()
        self._get_num_nodes_edges()

    def _get_num_nodes_edges(self):
        ''' Calculates the number of edges and nodes of the graph
            at current timestamp'''
        
        self.num_nodes = len(self.row_offset) - 1
        self.num_edges = len(self.column_indices) - self.column_indices.count(-1)