import copy
import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.static.csr import CSR, print_dev_array
from collections import deque
import time

class NaiveGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__(edge_list, max_num_nodes)
        # inspect(edge_list)
        self._prepare_edge_lst_fwd(edge_list)
        self._prepare_edge_lst_bwd(self.fwd_edge_list)  
        self._forward_graph = [CSR(self.fwd_edge_list[i], self.graph_updates[str(i)]["num_nodes"], is_edge_reverse=True) for i in range(len(self.fwd_edge_list))]
        self._backward_graph = [CSR(self.bwd_edge_list[i], self.graph_updates[str(i)]["num_nodes"]) for i in range(len(self.bwd_edge_list))]
        # self._forward_graph = [CSR(self.fwd_edge_list[0], self.graph_updates[str(0)]["num_nodes"], is_edge_reverse=True)]
        # self._backward_graph = [CSR(self.bwd_edge_list[0], self.graph_updates[str(0)]["num_nodes"])]

        self._get_graph_csr_ptrs(0)
        
    def _prepare_edge_lst_fwd(self, edge_list):   
        self.fwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = edge_list[i]
            edge_list_for_t.sort(key = lambda x: (x[1],x[0]))
            edge_list_for_t = [(edge_list_for_t[j][0],edge_list_for_t[j][1],j) for j in range(len(edge_list_for_t))]
            self.fwd_edge_list.append(edge_list_for_t)
    
    def _prepare_edge_lst_bwd(self, edge_list):    
        self.bwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = copy.deepcopy(edge_list[i])
            edge_list_for_t.sort()
            self.bwd_edge_list.append(edge_list_for_t)
        
    # def _graph_stack_push(self, elem):
    #     self.graph_stack.append(elem)
    
    # def _graph_stack_pop(self):
    #     self.graph_stack.pop()
        
    # def _graph_stack_top(self):
    #     return self.graph_stack[-1]
        
    def graph_type(self):
        return "csr"
        
    def in_degrees(self):
        return np.array(self._forward_graph[self.current_timestamp].out_degrees, dtype='int32')
    
    def out_degrees(self):
        return np.array(self._forward_graph[self.current_timestamp].in_degrees, dtype='int32')
    
    def _get_graph_csr_ptrs(self, timestamp):
        if self._is_backprop_state:
            bwd_csr_ptrs = self._backward_graph[timestamp]
            # bwd_csr_ptrs.print_csr_arrays()
            self.bwd_row_offset_ptr = bwd_csr_ptrs.row_offset_ptr
            self.bwd_column_indices_ptr = bwd_csr_ptrs.column_indices_ptr
            self.bwd_eids_ptr = bwd_csr_ptrs.eids_ptr
        else:
            fwd_csr_ptrs = self._forward_graph[timestamp]
            # fwd_csr_ptrs.print_csr_arrays()
            self.fwd_row_offset_ptr = fwd_csr_ptrs.row_offset_ptr
            self.fwd_column_indices_ptr = fwd_csr_ptrs.column_indices_ptr
            self.fwd_eids_ptr = fwd_csr_ptrs.eids_ptr

            # print(f"(PYTHON) RECEIVED ROW OFFSET PTR: {fwd_csr_ptrs.row_offset_ptr}")
            # print_dev_array(fwd_csr_ptrs.row_offset_ptr,129)
            # print("QUITTING")
            # quit()
    
    def _update_graph_forward(self):
        ''' Updates the current base graph to the next timestamp
        '''
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_forward()")
        self._get_graph_csr_ptrs(self.current_timestamp + 1)
        
    def _init_reverse_graph(self):
        ''' Generates the reverse of the base graph'''
        self._get_graph_csr_ptrs(self.current_timestamp)
        
    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception("⏰ Invalid timestamp during SeastarGraph.update_graph_backward()")
        self._get_graph_csr_ptrs(self.current_timestamp - 1)
