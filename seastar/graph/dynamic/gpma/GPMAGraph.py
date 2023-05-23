import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.gpma.gpma import GPMA, init_gpma, print_gpma_info, edge_update_list, label_edges, copy_label_edges, build_reverse_gpma, get_csr_ptrs, get_graph_attr


class GPMAGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__(edge_list, max_num_nodes)
        
        # forward and backward graphs for GPMA
        self._forward_graph = GPMA()
        self._backward_graph = GPMA()
        
        init_gpma(self._forward_graph, self.max_num_nodes)
        init_gpma(self._backward_graph, self.max_num_nodes)
        
        # initialise the graph for the first time stamp
        initial_graph_additions = self.graph_updates["0"]["add"]

        edge_update_list(self._forward_graph, initial_graph_additions, is_reverse_edge=True)
        label_edges(self._forward_graph)

        self._get_graph_csr_ptrs()
        
    def graph_type(self):
        return "gpma"
        
    def in_degrees(self):
        return np.array(self._forward_graph.out_degree, dtype='int32')
    
    def out_degrees(self):
        return np.array(self._forward_graph.in_degree, dtype='int32')
    
    def _update_reverse_graph_cache(self):
        # saving reverse base graph in cache
        self.graph_cache['reverse'] = copy.deepcopy(self._backward_graph)
            
    def _get_reverse_cached_graph(self):
        return copy.deepcopy(self.graph_cache['reverse'])
    
    def _get_graph_csr_ptrs(self):
        forward_csr_ptrs = get_csr_ptrs(self._forward_graph)
        backward_csr_ptrs = get_csr_ptrs(self._backward_graph)
        
        self.fwd_row_offset_ptr = forward_csr_ptrs[0]
        self.fwd_column_indices_ptr = forward_csr_ptrs[1]
        self.fwd_eids_ptr = forward_csr_ptrs[2]
        
        self.bwd_row_offset_ptr = backward_csr_ptrs[0]
        self.bwd_column_indices_ptr = backward_csr_ptrs[1]
        self.bwd_eids_ptr = backward_csr_ptrs[2]
    
    #TODO: Right now this returns (max_num_nodes,num_edges) see if this is what is required
    # def _get_graph_attributes(self):

    #     if not self._is_backprop_state:
    #         graph_attr = get_graph_attr(self._forward_graph)
    #     else:
    #         graph_attr = get_graph_attr(self._backward_graph)
        
    #     self.num_nodes = graph_attr[0]
    #     self.num_edges = graph_attr[1]
    
    def _update_graph_forward(self):
        
        # if we went through the entire time-stamps
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        # getting the graph edge modifications in 
        # the following form list[tuple(int, int)]
        graph_additions = self.graph_updates[str(self.current_timestamp + 1)]["add"]
        graph_deletions = self.graph_updates[str(self.current_timestamp + 1)]["delete"]

        edge_update_list(self._forward_graph, graph_additions, is_reverse_edge=True)
        edge_update_list(self._forward_graph, graph_deletions, is_delete=True, is_reverse_edge=True)

        # TODO: UNCOMMENT LATER
        label_edges(self._forward_graph)
        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()  # NOTE:
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'reverse' in self.graph_cache:
            self._backward_graph = self._get_reverse_cached_graph()
        else:
            build_reverse_gpma(self._backward_graph, self._forward_graph)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_reverse_graph_cache()

        self._get_graph_csr_ptrs()
        # self._get_graph_attributes() # NOTE:
        
    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_timestamp)]["add"]

        edge_update_list(self._backward_graph, graph_additions)
        edge_update_list(self._backward_graph, graph_deletions, is_delete=True)

        edge_update_list(self._forward_graph, graph_additions, is_reverse_edge=True)
        edge_update_list(self._forward_graph, graph_deletions, is_delete=True, is_reverse_edge=True)

        # TODO: UNCOMMENT LATER
        label_edges(self._forward_graph)
        copy_label_edges(self._backward_graph, self._forward_graph)

        self._get_graph_csr_ptrs()
        # self._get_graph_attributes()  # NOTE: