# TODO: add this import as well
from seastar.graph.SeastarGraph import SeastarGraph

import copy

from abc import ABC, abstractmethod

class DynamicGraph(SeastarGraph):
    def __init__(self, graph_updates, max_num_nodes):
        super().__init__()
        self.graph_updates = graph_updates
        self.max_num_nodes = max_num_nodes

        self.graph_cache = {}
        self._is_reverse_graph = False

        self.current_timestamp = 0
        
    def _update_graph_cache(self, is_reverse=False):
        # saving base graph in cache
        if not is_reverse:
            self.graph_cache['base'] = copy.deepcopy(self.forward_graph)
        else:
        # saving reverse base graph in cache
            self.graph_cache['reverse'] = copy.deepcopy(self.backward_graph)
            
    def _get_cached_graph(self, is_reverse=False):
        if not is_reverse:
            return copy.deepcopy(self.graph_cache['base'])
        else:
            return copy.deepcopy(self.graph_cache['reverse'])
    
    def _revert_to_base_graph(self):
        self.forward_graph = self._get_cached_graph(is_reverse=False)

        self._get_graph_csr_ptrs()
        self._is_reverse_graph = False
        
    def get_graph(self, timestamp: int):

        if self._is_reverse_graph:
            self._revert_to_base_graph()
        
        if timestamp < self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        while self.current_timestamp < timestamp:
            self._update_graph_forward()

    def get_backward_graph(self, timestamp: int):

        if not self._is_reverse_graph:
            self._init_reverse_graph()
        
        if timestamp > self.current_timestamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        while self.current_timestamp > timestamp:
            self._update_graph_backward()
    
    def _get_graph_csr_ptrs(self):
        if not self._is_reverse_graph:
            csr_ptrs = self.forward_graph.get_csr_ptrs()
        else:
            csr_ptrs = self.backward_graph.get_csr_ptrs()
        
        self.row_offset_ptr = csr_ptrs[0]
        self.column_indices_ptr = csr_ptrs[1]
        self.eids_ptr = csr_ptrs[2]
    
    @abstractmethod
    def in_degrees(self):
        pass
    
    @abstractmethod
    def out_degrees(self):
        pass
    
    
    @abstractmethod
    def _update_graph_forward(self):
        pass
    
    @abstractmethod
    def _init_reverse_graph(self):
        pass
    
    @abstractmethod
    def _update_graph_backward(self):
        pass
    
    