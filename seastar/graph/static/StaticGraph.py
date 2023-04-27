# TODO: add this import as well
from seastar.graph.SeastarGraph import SeastarGraph

import copy
from abc import ABC, abstractmethod

from seastar.graph.static.csr import CSR

class StaticGraph(SeastarGraph):
    def __init__(self, edge_list, num_nodes):    
        super().__init__()
        self.forward_graph = CSR(edge_list, num_nodes)
        self.backward_graph = CSR(edge_list, num_nodes, is_edge_reverse=True)
        
        # TODO: Add these
        # self._get_graph_csr_ptrs()