import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.pcsr.pcsr import PCSR, copy_label_edges, build_reverse_pcsr

class PCSRGraph(DynamicGraph):
    def __init__(self, edge_list):
        super().__init__(edge_list)
        
        self._forward_graph = PCSR(self.max_num_nodes)
        self._backward_graph = PCSR(self.max_num_nodes)
        self._forward_graph.edge_update_list(self.graph_updates["0"]["add"],is_reverse_edge=True)

        self._forward_graph.label_edges()
        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()
        self._update_graph_cache()  # saving the base graph in cache
        
    def graph_type(self):
        return "pcsr"
        
    def in_degrees(self):
        return np.array([node.num_neighbors for node in self._forward_graph.nodes], dtype='int32')
    
    def out_degrees(self):
        return np.array([node.in_degree for node in self._forward_graph.nodes], dtype='int32')
    
    def _get_graph_csr_ptrs(self):
        forward_csr_ptrs = self._forward_graph.get_csr_ptrs()
        backward_csr_ptrs = self._backward_graph.get_csr_ptrs()
        
        self.fwd_row_offset_ptr = forward_csr_ptrs[0]
        self.fwd_column_indices_ptr = forward_csr_ptrs[1]
        self.fwd_eids_ptr = forward_csr_ptrs[2]
        
        self.bwd_row_offset_ptr = backward_csr_ptrs[0]
        self.bwd_column_indices_ptr = backward_csr_ptrs[1]
        self.bwd_eids_ptr = backward_csr_ptrs[2]

    # def _get_graph_attributes(self):
    #     if not self._is_backprop_state:
    #         graph_attr = self._forward_graph.get_graph_attr()
    #     else:
    #         graph_attr = self._backward_graph.get_graph_attr()
        
    #     self.num_nodes = graph_attr[0]
    #     self.num_edges = graph_attr[1]
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''

        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp + 1)]["add"]
        graph_deletions = self.graph_updates[str(self.current_timestamp + 1)]["delete"]

        self._forward_graph.edge_update_list(graph_additions,is_reverse_edge=True)
        self._forward_graph.edge_update_list(graph_deletions,is_delete=True,is_reverse_edge=True)

        self._forward_graph.label_edges()
        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'reverse' in self.graph_cache:
            self._backward_graph = self._get_cached_graph(is_reverse=True)
        else:
            build_reverse_pcsr(self._backward_graph, self._forward_graph)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_graph_cache(is_reverse=True)

        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()
        
    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_timestamp)]["add"]

        self._backward_graph.edge_update_list(graph_additions)
        self._forward_graph.edge_update_list(graph_additions,is_reverse_edge=True)
        
        self._backward_graph.edge_update_list(graph_deletions,is_delete=True)
        self._forward_graph.edge_update_list(graph_deletions, is_delete=True, is_reverse_edge=True)

        self._forward_graph.label_edges()
        copy_label_edges(self._backward_graph, self._forward_graph)
        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()
