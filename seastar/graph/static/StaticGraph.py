from abc import ABC, abstractmethod
import copy

import numpy as np

from rich.console import Console

console = Console()

from seastar.graph.SeastarGraph import SeastarGraph


from seastar.graph.static.csr import CSR

class StaticGraph(SeastarGraph):
    def __init__(self, edge_list, num_nodes):    
        super().__init__()
        self._num_nodes = num_nodes
        self._num_edges = len(set(edge_list))
        
        # console.log("Building forward edge list")
        self._prepare_edge_lst_fwd(edge_list)
        # console.log("Creating forward graph")
        self._forward_graph = CSR(self.fwd_edge_list, self._num_nodes, is_edge_reverse=True)
        
        # console.log("Building backward edge list")
        self._prepare_edge_lst_bwd(self.fwd_edge_list)
        # console.log("Creating backward graph")
        self._backward_graph = CSR(self.bwd_edge_list, self._num_nodes)
        
        # console.log("Getting CSR ptrs")
        self._get_graph_csr_ptrs()
    
    def _prepare_edge_lst_fwd(self, edge_list):    
        edge_list_for_t = edge_list
        edge_list_for_t.sort(key = lambda x: (x[1],x[0]))
        edge_list_for_t = [(edge_list_for_t[j][0],edge_list_for_t[j][1],j) for j in range(len(edge_list_for_t))]
        self.fwd_edge_list = edge_list_for_t
    
    def _prepare_edge_lst_bwd(self, edge_list):    
        edge_list_for_t = copy.deepcopy(edge_list)
        edge_list_for_t.sort()
        self.bwd_edge_list = edge_list_for_t
        
    def _get_graph_csr_ptrs(self):
        fwd_csr_ptrs = self._forward_graph
        self.fwd_row_offset_ptr = fwd_csr_ptrs.row_offset_ptr
        self.fwd_column_indices_ptr = fwd_csr_ptrs.column_indices_ptr
        self.fwd_eids_ptr = fwd_csr_ptrs.eids_ptr
        
        bwd_csr_ptrs = self._backward_graph
        self.bwd_row_offset_ptr = bwd_csr_ptrs.row_offset_ptr
        self.bwd_column_indices_ptr = bwd_csr_ptrs.column_indices_ptr
        self.bwd_eids_ptr = bwd_csr_ptrs.eids_ptr
            
    def get_num_nodes(self):
        return self._num_nodes
    
    def get_num_edges(self):
        return self._num_edges
        
    def graph_type(self):
        return "csr"
    
    def in_degrees(self):
        return np.array(self._forward_graph.out_degrees, dtype='int32')
    
    def out_degrees(self):
        return np.array(self._forward_graph.in_degrees, dtype='int32')