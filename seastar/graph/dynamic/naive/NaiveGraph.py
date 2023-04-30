import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.static.csr import CSR
from collections import deque

class NaiveGraph(DynamicGraph):
    def __init__(self, edge_list):
        super().__init__(edge_list)
        
        self.graph_stack = deque()
        self.edge_list = edge_list
        
        self._forward_graph = CSR(edge_list[0], self.graph_updates["0"]["num_nodes"], is_edge_reverse=True)
        self._forward_graph.label_edges()

        self._graph_stack_push(self._forward_graph)
        self._get_graph_csr_ptrs()
    
    def _graph_stack_push(self, elem):
        self.graph_stack.append(elem)
    
    def _graph_stack_pop(self):
        elem = self.graph_stack[-1]
        return elem
        
    def graph_type(self):
        return "csr"
        
    def in_degrees(self):
        return np.array(self._forward_graph.out_degrees, dtype='int32')
    
    def out_degrees(self):
        return np.array(self._forward_graph.in_degrees, dtype='int32')
    
    def _get_graph_csr_ptrs(self):
        fwd_csr_ptrs = self._forward_graph.get_csr_ptrs()
        self.fwd_row_offset_ptr = fwd_csr_ptrs[0]
        self.fwd_column_indices_ptr = fwd_csr_ptrs[1]
        self.fwd_eids_ptr = fwd_csr_ptrs[2]
        
        if self._backward_graph is not None:
            bwd_csr_ptrs = self._backward_graph.get_csr_ptrs()
            self.bwd_row_offset_ptr = bwd_csr_ptrs[0]
            self.bwd_column_indices_ptr = bwd_csr_ptrs[1]
            self.bwd_eids_ptr = bwd_csr_ptrs[2]
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''

        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        self._forward_graph = CSR(self.edge_list[self.current_timestamp], self.graph_updates[str(self.current_timestamp+1)]["num_nodes"], is_edge_reverse=True)
        self._forward_graph.label_edges()
        self._graph_stack_push(self._forward_graph)
        self._get_graph_csr_ptrs()
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        fwd_graph = self._graph_stack_pop()
        self._backward_graph = CSR(self.edge_list[self.current_timestamp], self.graph_updates[str(self.current_timestamp)]["num_nodes"])
        self._backward_graph.copy_label_edges(fwd_graph) 
        self._get_graph_csr_ptrs()
        
    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        fwd_graph = self._graph_stack_pop()
        self._backward_graph = CSR(self.edge_list[self.current_timestamp - 1], self.graph_updates[str(self.current_timestamp - 1)]["num_nodes"])
        self._backward_graph.copy_label_edges(fwd_graph) 
        self._get_graph_csr_ptrs()
