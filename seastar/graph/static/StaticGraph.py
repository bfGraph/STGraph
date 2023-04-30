from abc import ABC, abstractmethod
import copy

import numpy as np

from seastar.graph.SeastarGraph import SeastarGraph


from seastar.graph.static.csr import CSR

class StaticGraph(SeastarGraph):
    def __init__(self, edge_list, num_nodes):    
        super().__init__()
        self.forward_graph = CSR(edge_list, num_nodes, is_edge_reverse=True)
        self.backward_graph = CSR(edge_list, num_nodes)
        self.forward_graph.label_edges()
        self.backward_graph.copy_label_edges(self.forward_graph)
        
        self.forward_graph.print_graph()
        
        self.backward_graph.print_graph()
        
        
        self._get_graph_csr_ptrs()
        
    def _get_graph_csr_ptrs(self):
        fwd_csr_ptrs = self.forward_graph.get_csr_ptrs()
        self.fwd_row_offset_ptr = fwd_csr_ptrs[0]
        self.fwd_column_indices_ptr = fwd_csr_ptrs[1]
        self.fwd_eids_ptr = fwd_csr_ptrs[2]
        
        bwd_csr_ptrs = self.backward_graph.get_csr_ptrs()
        self.bwd_row_offset_ptr = bwd_csr_ptrs[0]
        self.bwd_column_indices_ptr = bwd_csr_ptrs[1]
        self.bwd_eids_ptr = bwd_csr_ptrs[2]
        
        
    def graph_type(self):
        return "csr"
    
    def in_degrees(self):
        return np.array(self.forward_graph.out_degrees, dtype='int32')
    
    def out_degrees(self):
        return np.array(self.forward_graph.in_degrees, dtype='int32')