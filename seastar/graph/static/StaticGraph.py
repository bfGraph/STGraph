# TODO: add this import as well
# from SeastarGraph import SeastarGraph

import copy

from abc import ABC, abstractmethod

# TODO: add SeastarGraph class as part of multiple inheritance
class StaticGraph(ABC):
    def __init__(self, edge_list):
        # Could possibly move to SeastarGraph class
        self.ndata = {}
        
        # Could possibly move to SeastarGraph class
        self.num_nodes = 0
        self.num_edges = 0
        
        # Could possibly move to SeastarGraph class
        self.row_offset_ptr = None
        self.column_indices_ptr = None
        self.eids_ptr = None
        
        # TODO: Add these
        # self.graph = CSR(edge_list)
        # self._get_graph_csr_ptrs()