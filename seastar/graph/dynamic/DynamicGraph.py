# TODO: add this import as well
# from SeastarGraph import SeastarGraph

import copy

from abc import ABC, abstractmethod

# TODO: add SeastarGraph class as part of multiple inheritance
class DynamicGraph(ABC):
    def __init__(self, graph_updates, max_num_nodes):
        self.graph_updates = graph_updates
        self.max_num_nodes = max_num_nodes

        self.graph_cache = {}
        self._is_reverse_graph = False
        self.ndata = {}     # Could possibly move to SeastarGraph class

        self.current_time_stamp = 0

        # Could possibly move to SeastarGraph class
        self.num_nodes = 0
        self.num_edges = 0

        # Could possibly move to SeastarGraph class
        self.row_offset_ptr = None
        self.column_indices_ptr = None
        self.eids_ptr = None
        
    def _update_graph_cache(self, is_reverse=True):
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
        
    def get_forward_graph_for_timestamp(self, timestamp: int):

        if self._is_reverse_graph:
            self._revert_to_base_graph()
        
        if timestamp < self.current_time_stamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        while self.current_time_stamp < timestamp:
            self._update_graph_forward()

    def get_backward_graph_for_timestamp(self, timestamp: int):

        if not self._is_reverse_graph:
            self._init_reverse_graph()
        
        if timestamp > self.current_time_stamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        while self.current_time_stamp > timestamp:
            self._update_graph_backward()
    
    @abstractmethod
    def in_degrees(self):
        pass
    
    @abstractmethod
    def out_degrees(self):
        pass
    
    @abstractmethod
    def _get_graph_csr_ptrs(self):
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
    
    