import numpy as np
import copy

from gpma import GPMA, init_gpma, print_gpma_info, edge_update_list, label_edges, copy_label_edges, build_reverse_gpma, get_csr_ptrs, get_graph_attr

class GPMAGraph:
    def __init__(self, graph_updates: dict, max_num_nodes: int):
        self.max_num_nodes = max_num_nodes
        self.graph_cache = {}

        self._forward_graph = GPMA()
        self._backward_graph = GPMA()
        
        init_gpma(self._forward_graph, max_num_nodes)
        init_gpma(self._backward_graph, max_num_nodes)

        self._is_backprop_state = False

        self.graph_updates = graph_updates
        self.ndata = {}
        self.current_time_stamp = 0
        self.num_nodes = 0
        self.num_edges = 0

        # pointer to thrust device vectors
        self.row_offset_ptr = None
        self.column_indices_ptr = None
        self.eids_ptr = None

        # initial_graph_additions must be a list of edge tuples
        # list[tuple(int, int)]
        initial_graph_additions = graph_updates["0"]["add"]

        edge_update_list(self._forward_graph, initial_graph_additions, is_reverse_edge=True)
        label_edges(self._forward_graph)

        self._get_graph_csr_ptrs()
        self._get_graph_attributes()
        self._update_graph_cache()


    def _update_graph_cache(self, is_reverse=False):
        ''' Sets the base graph cache with the current base graph
        
            The base graph and it's CSR arrays are stored in a dictionary
            for faster retrieval. Instead of creating the base graph and
            it's CSR arrays, it can be retrieved from the cache.
            Contains the base_graph, row_offset, column_indices, eids,
            num_nodes and num_edges
        '''
        # saving base graph in cache
        if not is_reverse:
            self.graph_cache['base'] = copy.deepcopy(self._forward_graph)
        else:
        # saving reverse base graph in cache
            self.graph_cache['reverse'] = copy.deepcopy(self._backward_graph)

    def _get_cached_graph(self, is_reverse=False):
        if not is_reverse:
            return copy.deepcopy(self.graph_cache['base'])
        else:
            return copy.deepcopy(self.graph_cache['reverse'])

    def in_degrees(self):
        return np.array(self._forward_graph.out_degree, dtype='int32')
    
    # TODO:
    def _get_graph_csr_ptrs(self):

        if not self._is_backprop_state:
            csr_ptrs = get_csr_ptrs(self._forward_graph)
        else:
            csr_ptrs = get_csr_ptrs(self._backward_graph)

        self.row_offset_ptr = csr_ptrs[0]
        self.column_indices_ptr = csr_ptrs[1]
        self.eids_ptr = csr_ptrs[2]
    
    def _get_graph_attributes(self):

        if not self._is_backprop_state:
            graph_attr = get_graph_attr(self._forward_graph)
        else:
            graph_attr = get_graph_attr(self._backward_graph)
        
        self.num_nodes = graph_attr[0]
        self.num_edges = graph_attr[1]


    def _update_graph_forward(self):

        # if we went through the entire time-stamps
        if str(self.current_time_stamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        self.current_time_stamp += 1
        
        # getting the graph edge modifications in 
        # the following form list[tuple(int, int)]
        graph_additions = self.graph_updates[str(self.current_time_stamp)]["add"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp)]["delete"]

        edge_update_list(self._forward_graph, graph_additions, is_reverse_edge=True)
        edge_update_list(self._forward_graph, graph_deletions, is_delete=True, is_reverse_edge=True)

        label_edges(self._forward_graph)
        self._get_graph_csr_ptrs()
        self._get_graph_attributes()

    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''

        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'reverse' in self.graph_cache:
            self._backward_graph = self._get_cached_graph(is_reverse=True)
        else:
            build_reverse_gpma(self._backward_graph, self._forward_graph)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_graph_cache(is_reverse=True)

        self._is_backprop_state = True

        self._get_graph_csr_ptrs()
        self._get_graph_attributes()

    def _update_graph_backward(self):
        if self.current_time_stamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        self.current_time_stamp -= 1
        
        graph_additions = self.graph_updates[str(self.current_time_stamp + 1)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_time_stamp + 1)]["add"]

        edge_update_list(self._backward_graph, graph_additions)
        edge_update_list(self._backward_graph, graph_deletions, is_delete=True)

        edge_update_list(self._forward_graph, graph_additions, is_reverse_edge=True)
        edge_update_list(self._forward_graph, graph_deletions, is_delete=True, is_reverse_edge=True)

        label_edges(self._forward_graph)
        copy_label_edges(self._backward_graph, self._forward_graph)

        self._get_graph_csr_ptrs()
        self._get_graph_attributes()

    def get_forward_graph_for_timestamp(self, timestamp: int):

        if self._is_backprop_state:
            self._revert_to_base_graph()
        
        if timestamp < self.current_time_stamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")

        while self.current_time_stamp < timestamp:
            self._update_graph_forward()

    def get_backward_graph_for_timestamp(self, timestamp: int):

        if not self._is_backprop_state:
            self._init_reverse_graph()
        
        if timestamp > self.current_time_stamp:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        while self.current_time_stamp > timestamp:
            self._update_graph_backward()