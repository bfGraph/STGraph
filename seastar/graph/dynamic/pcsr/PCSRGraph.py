import copy
import numpy as np
from rich import inspect
import time

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.pcsr.pcsr import PCSR

class PCSRGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__(edge_list, max_num_nodes)
        
        # Get the maximum number of edges
        self._get_max_num_edges()
        
        self._forward_graph = PCSR(self.max_num_nodes, self.max_num_edges)
        self._forward_graph.edge_update_list(self.graph_updates["0"]["add"], is_reverse_edge=True)
        self._forward_graph.label_edges()   
        self._forward_graph.build_csr()
        self._get_graph_csr_ptrs()
        
        # Using cache to make pcsr faster
        self.graph_cache = {}
        self._in_degrees_cache = {}
        self._out_degrees_cache = {}
    
    def _get_max_num_edges(self):
        updates = self.graph_updates
        edge_set = set()
        for i in range(len(updates)):
          for j in range(len(updates[str(i)]["add"])):
            edge_set.add(updates[str(i)]["add"][j])
        self.max_num_edges = len(edge_set)
                
    def graph_type(self):
        return "pcsr"
        
    def in_degrees(self):
        if self.current_timestamp not in self._in_degrees_cache:
            self._in_degrees_cache[self.current_timestamp] = np.array(self._forward_graph.out_degrees, dtype='int32')
        
        return self._in_degrees_cache[self.current_timestamp]
    
    def out_degrees(self):
        if self.current_timestamp not in self._out_degrees_cache:
            self._out_degrees_cache[self.current_timestamp] = np.array(self._forward_graph.in_degrees, dtype='int32')
        
        return self._out_degrees_cache[self.current_timestamp]
    
    def _get_graph_csr_ptrs(self): 
        csr_ptrs = self._forward_graph.get_csr_ptrs()
        if self._is_backprop_state:
            self.bwd_row_offset_ptr = csr_ptrs[0]
            self.bwd_column_indices_ptr = csr_ptrs[1]
            self.bwd_eids_ptr = csr_ptrs[2]
        else:
            self.fwd_row_offset_ptr = csr_ptrs[0]
            self.fwd_column_indices_ptr = csr_ptrs[1]
            self.fwd_eids_ptr = csr_ptrs[2]
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp + 1)]["add"]
        graph_deletions = self.graph_updates[str(self.current_timestamp + 1)]["delete"]

        self._forward_graph.edge_update_list(graph_additions, is_reverse_edge=True)
        self._forward_graph.edge_update_list(graph_deletions, is_delete=True, is_reverse_edge=True)
        self._forward_graph.label_edges()   
        self._forward_graph.build_csr()
        self._get_graph_csr_ptrs()
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''
        self._forward_graph.build_reverse_csr()
        self._get_graph_csr_ptrs()

    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_timestamp)]["add"]

        self._forward_graph.edge_update_list(graph_additions, is_reverse_edge=True)   
        self._forward_graph.edge_update_list(graph_deletions, is_delete=True, is_reverse_edge=True)
        self._forward_graph.label_edges() 
        self._forward_graph.build_reverse_csr()
        self._get_graph_csr_ptrs()