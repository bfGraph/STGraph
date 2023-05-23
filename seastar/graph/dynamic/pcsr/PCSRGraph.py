import copy
import numpy as np
from rich import inspect
import time

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.pcsr.pcsr import PCSR, build_reverse_pcsr

class PCSRGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__(edge_list, max_num_nodes)
        
        # preprocessing the edge ids for forward and backward pass
        # so that it doesn't have to be calculated each time
        self._prepare_eid_bwd(edge_list) 
        
        self._forward_graph = PCSR(self.max_num_nodes)
        self._backward_graph = PCSR(self.max_num_nodes)
        self._forward_graph.edge_update_list(self.graph_updates["0"]["add"],is_reverse_edge=True)
        
        self._update_graph_cache()
        self._get_graph_csr_ptrs(eids=list())
    
    def _prepare_eid_bwd(self, edge_list):   
        fwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = edge_list[i]
            edge_list_for_t.sort(key = lambda x: (x[1],x[0]))
            edge_list_for_t = [(edge_list_for_t[j][0],edge_list_for_t[j][1],j) for j in range(len(edge_list_for_t))]
            fwd_edge_list.append(edge_list_for_t)
        
        self.bwd_eid_list = []
        for i in range(len(fwd_edge_list)):
            edge_list_for_t = fwd_edge_list[i]
            edge_list_for_t.sort()
            eid_t = [edge_list_for_t[i][2] for i in range(len(edge_list_for_t))]
            self.bwd_eid_list.append(eid_t)
                
    def graph_type(self):
        return "pcsr"
        
    def in_degrees(self):
        return np.array([node.num_neighbors for node in self._forward_graph.nodes], dtype='int32')
    
    def out_degrees(self):
        return np.array([node.in_degree for node in self._forward_graph.nodes], dtype='int32')
    
    def _update_graph_cache(self, is_bwd=False):
        if is_bwd:
            # saving reverse base graph in cache
            self.graph_cache['bwd'] = copy.deepcopy(self._backward_graph)
        else:
            self.graph_cache['fwd'] = copy.deepcopy(self._forward_graph)
            
    def _get_cached_graph(self, is_bwd=False):
        if is_bwd:
            return copy.deepcopy(self.graph_cache['bwd'])
        else:
            return copy.deepcopy(self.graph_cache['fwd'])
    
    def _get_graph_csr_ptrs(self, eids): 

        if self._is_backprop_state:
            backward_csr_ptrs = self._backward_graph.get_csr_ptrs(eids=eids)
            self.bwd_row_offset_ptr = backward_csr_ptrs[0]
            self.bwd_column_indices_ptr = backward_csr_ptrs[1]
            self.bwd_eids_ptr = backward_csr_ptrs[2]
        else:
            forward_csr_ptrs = self._forward_graph.get_csr_ptrs(eids=eids)
            self.fwd_row_offset_ptr = forward_csr_ptrs[0]
            self.fwd_column_indices_ptr = forward_csr_ptrs[1]
            self.fwd_eids_ptr = forward_csr_ptrs[2]
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp + 1)]["add"]
        graph_deletions = self.graph_updates[str(self.current_timestamp + 1)]["delete"]
        
        # print(f"(FORWARD) ➕➕➕ Adding a total of {len(graph_additions)} elements", flush=True)
        # print(f"(FORWARD) ➖➖➖ Removing a total of {len(graph_deletions)} elements", flush=True)

        self._forward_graph.edge_update_list(graph_additions,is_reverse_edge=True)
        self._forward_graph.edge_update_list(graph_deletions,is_delete=True,is_reverse_edge=True)
        self._get_graph_csr_ptrs(eids=list())
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''
        
        # checking if the reverse base graph exists in the cache
        # we can load it from there instead of building it each time
        if 'bwd' in self.graph_cache:
            self._backward_graph = self._get_cached_graph(is_bwd=True)
        else:
            build_reverse_pcsr(self._backward_graph, self._forward_graph)

            # storing the reverse base graph in cache after building
            # it for the first time
            self._update_graph_cache(is_bwd=True)
            
        self._get_graph_csr_ptrs(eids=self.bwd_eid_list[self.current_timestamp])
        
        # Resetting the forward graph so that it can be used later
        self._forward_graph = self._get_cached_graph()
        
    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        
        graph_additions = self.graph_updates[str(self.current_timestamp)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_timestamp)]["add"]
        
        # print(f"(BACKWARD) ➕➕➕ Adding a total of {len(graph_additions)} elements",flush=True)
        # print(f"(BACKWARD) ➖➖➖ Removing a total of {len(graph_deletions)} elements",flush=True)
        
        self._backward_graph.edge_update_list(graph_additions)   
        self._backward_graph.edge_update_list(graph_deletions,is_delete=True)
        self._get_graph_csr_ptrs(eids=self.bwd_eid_list[self.current_timestamp - 1])