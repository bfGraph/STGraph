import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.pcsr.pcsr import PCSR

class PCSRGraph(DynamicGraph):
    def __init__(self, graph_updates, max_num_nodes):
        super().__init__(graph_updates, max_num_nodes)
        
        self.forward_graph = PCSR(max_num_nodes)
        self.backward_graph = PCSR(max_num_nodes)
        
        initial_graph_additions = graph_updates["0"]["add"]

        for edge in initial_graph_additions:
            self.forward_graph.add_edge(edge[1], edge[0], 1)

        self._get_graph_csr_ptrs()
        self._update_graph_cache()  # saving the base graph in cache
        
    def in_degrees(self):
        return np.array([node.num_neighbors for node in self.forward_graph.nodes], dtype='int32')
    
    def out_degrees(self):
        return np.array([node.in_degree for node in self.forward_graph.nodes], dtype='int32')
    
    # TODO: Need to figure out the GPU pointer bug
    def _get_graph_csr_ptrs(self):
        pass   
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        
        '''

        if str(self.current_time_stamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        self.current_time_stamp += 1
        
        graph_additions = self.graph_updates[str(self.current_time_stamp)]["add"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp)]["delete"]

        for edge in graph_additions:
            self.forward_graph.add_edge(edge[1], edge[0], 1)

        for edge in graph_deletions:
            self.forward_graph.delete_edge(edge[1], edge[0])

        self._get_graph_csr_ptrs()
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'reverse' in self.graph_cache:
            self.backward_graph = self._get_cached_graph(is_reverse=True)
        else:
            edges = self.forward_graph.get_edges()
            for src, dst, val in edges:
                self.backward_graph.add_edge(dst, src, val)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_graph_cache(is_reverse=True)

        self._is_reverse_graph = True

        self._get_graph_csr_ptrs()
        
    def _update_graph_backward(self):
        if self.current_time_stamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        self.current_time_stamp -= 1
        
        graph_additions = self.graph_updates[str(self.current_time_stamp + 1)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp + 1)]["add"]

        for edge in graph_additions:
            self.backward_graph.add_edge(edge[0], edge[1], 1)
            self.forward_graph.add_edge(edge[1], edge[0], 1)

        for edge in graph_deletions:
            self.backward_graph.delete_edge(edge[0], edge[1])
            self.forward_graph.delete_edge(edge[1], edge[0])

        self._get_graph_csr()
