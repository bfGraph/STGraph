import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.pcsr.pcsr import PCSR, copy_label_edges, build_reverse_pcsr

class PCSRGraph(DynamicGraph):
    def __init__(self, graph_updates, max_num_nodes):
        super().__init__(graph_updates, max_num_nodes)
        
        self.forward_graph = PCSR(max_num_nodes)
        self.backward_graph = PCSR(max_num_nodes)
        self.forward_graph.edge_update_list(graph_updates["0"]["add"],is_reverse_edge=True)

        self.forward_graph.label_edges()
        self._get_graph_csr_ptrs()
        self._get_graph_attributes()
        self._update_graph_cache()  # saving the base graph in cache
        
    def in_degrees(self):
        return np.array([node.num_neighbors for node in self.forward_graph.nodes], dtype='int32')
    
    def out_degrees(self):
        return np.array([node.in_degree for node in self.forward_graph.nodes], dtype='int32')
    
    def _get_graph_csr_ptrs(self):
        if not self._is_reverse_graph:
            csr_ptrs = self.forward_graph.get_csr_ptrs()
        else:
            csr_ptrs = self.backward_graph.get_csr_ptrs()
        
        self.row_offset_ptr = csr_ptrs[0]
        self.column_indices_ptr = csr_ptrs[1]
        self.eids_ptr = csr_ptrs[2]

    def _get_graph_attributes(self):
        if not self._is_reverse_graph:
            graph_attr = self.forward_graph.get_graph_attr()
        else:
            graph_attr = self.backward_graph.get_graph_attr()
        
        self.num_nodes = graph_attr[0]
        self.num_edges = graph_attr[1]
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''

        if str(self.current_time_stamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        self.current_time_stamp += 1
        
        graph_additions = self.graph_updates[str(self.current_time_stamp)]["add"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp)]["delete"]

        self.forward_graph.edge_update_list(graph_additions,is_reverse_edge=True)
        self.forward_graph.edge_update_list(graph_deletions,is_delete=True,is_reverse_edge=True)

        self.forward_graph.label_edges()
        self._get_graph_csr_ptrs()
        self._get_graph_attributes()
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'reverse' in self.graph_cache:
            self.backward_graph = self._get_cached_graph(is_reverse=True)
        else:
            build_reverse_pcsr(self.backward_graph, self.forward_graph)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_graph_cache(is_reverse=True)

        self._is_reverse_graph = True
        self._get_graph_csr_ptrs()
        self._get_graph_attributes()
        
    def _update_graph_backward(self):
        if self.current_time_stamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        self.current_time_stamp -= 1
        
        graph_additions = self.graph_updates[str(self.current_time_stamp + 1)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp + 1)]["add"]

        self.backward_graph.edge_update_list(graph_additions)
        self.forward_graph.edge_update_list(graph_additions,is_reverse_edge=True)
        
        self.backward_graph.edge_update_list(graph_deletions,is_delete=True)
        self.forward_graph.edge_update_list(graph_deletions, is_delete=True, is_reverse_edge=True)

        self.forward_graph.label_edges()
        copy_label_edges(self.backward_graph, self.forward_graph)
        self._get_graph_csr_ptrs()
        self._get_graph_attributes()
