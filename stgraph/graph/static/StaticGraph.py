from abc import ABC, abstractmethod
import copy

import numpy as np

from rich.console import Console

console = Console()

from stgraph.graph.SeastarGraph import SeastarGraph


from seastar.graph.static.csr import CSR

class StaticGraph(SeastarGraph):
    def __init__(self, edge_list, edge_weights, num_nodes):    
        super().__init__()
        self._num_nodes = num_nodes
        self._num_edges = len(set(edge_list))
        
        # console.log("Building forward edge list")
        self._prepare_edge_lst_fwd(edge_list)
        # console.log("Creating forward graph")
        self._forward_graph = CSR(self.fwd_edge_list, edge_weights, self._num_nodes, is_edge_reverse=True)
        
        # console.log("Building backward edge list")
        self._prepare_edge_lst_bwd(self.fwd_edge_list)
        # console.log("Creating backward graph")
        self._backward_graph = CSR(self.bwd_edge_list, edge_weights, self._num_nodes)
        
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
        self.fwd_row_offset_ptr = self._forward_graph.row_offset_ptr
        self.fwd_column_indices_ptr = self._forward_graph.column_indices_ptr
        self.fwd_eids_ptr = self._forward_graph.eids_ptr
        self.fwd_node_ids_ptr = self._forward_graph.node_ids_ptr

        self.bwd_row_offset_ptr = self._backward_graph.row_offset_ptr
        self.bwd_column_indices_ptr = self._backward_graph.column_indices_ptr
        self.bwd_eids_ptr = self._backward_graph.eids_ptr
        self.bwd_node_ids_ptr = self._backward_graph.node_ids_ptr
            
    def get_num_nodes(self):
        return self._num_nodes
    
    def get_num_edges(self):
        return self._num_edges
    
    def get_ndata(self, field):
        if field in self._ndata:
            return self._ndata[field]
        else:
            return None

    def set_ndata(self, field, val):
        self._ndata[field] = val
        
    def graph_type(self):
        # return "csr"
        return "csr_unsorted"
    
    def in_degrees(self):
        return np.array(self._forward_graph.out_degrees, dtype='int32')
    
    def out_degrees(self):
        return np.array(self._forward_graph.in_degrees, dtype='int32')
    
    def weighted_in_degrees(self):
        return np.array(self._forward_graph.weighted_out_degrees, dtype='int32')