import copy
import numpy as np
from rich import inspect

from stgraph.graph.dynamic.DynamicGraph import DynamicGraph
from stgraph.graph.static.csr import CSR
from collections import deque
import time


class NaiveGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes, snapshot_edge_list, total_timestamps):
        super().__init__(edge_list, max_num_nodes, snapshot_edge_list, total_timestamps)

        self._prepare_edge_lst_fwd(edge_list)
        self._prepare_edge_lst_bwd(self.fwd_edge_list)

        # This is used because edge weights is a compulsary argument to CSR
        edge_weight_lst = [[1 for _ in edge_list_t] for edge_list_t in edge_list]

        self._forward_graph = [
            CSR(
                self.fwd_edge_list[i],
                edge_weight_lst[i],
                self.graph_attr[str(i)][0],
                is_edge_reverse=True,
            )
            for i in range(len(self.fwd_edge_list))
        ]
        self._backward_graph = [
            CSR(self.bwd_edge_list[i], edge_weight_lst[i], self.graph_attr[str(i)][0])
            for i in range(len(self.bwd_edge_list))
        ]

        # for benchmarking purposes
        self._update_count = 0
        self._total_update_time = 0
        self._gpu_move_time = 0

        self._get_graph_csr_ptrs(0)

    def _prepare_edge_lst_fwd(self, edge_list):
        self.fwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = edge_list[i]
            edge_list_for_t.sort(key=lambda x: (x[1], x[0]))
            edge_list_for_t = [
                (edge_list_for_t[j][0], edge_list_for_t[j][1], j)
                for j in range(len(edge_list_for_t))
            ]
            self.fwd_edge_list.append(edge_list_for_t)

    def _prepare_edge_lst_bwd(self, edge_list):
        self.bwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = copy.deepcopy(edge_list[i])
            edge_list_for_t.sort()
            self.bwd_edge_list.append(edge_list_for_t)

    def graph_type(self):
        return "csr"
    
    def _cache_graph(self):
        pass

    def _get_cached_graph(self, timestamp):
        return False

    def in_degrees(self):
        return np.array(self._forward_graph[self.current_timestamp].out_degrees, dtype="int32")
    
    def out_degrees(self):
        return np.array(self._forward_graph[self.current_timestamp].in_degrees, dtype="int32")
    
    # def weighted_in_degrees(self):
    #     return np.array(self._forward_graph[self.current_timestamp].weighted_out_degrees, dtype="float32")

    def _get_graph_csr_ptrs(self, timestamp):
        if self._is_backprop_state:
            bwd_csr_ptrs = self._backward_graph[timestamp]
            self.bwd_row_offset_ptr = bwd_csr_ptrs.row_offset_ptr
            self.bwd_column_indices_ptr = bwd_csr_ptrs.column_indices_ptr
            self.bwd_eids_ptr = bwd_csr_ptrs.eids_ptr
            self.bwd_node_ids_ptr = bwd_csr_ptrs.node_ids_ptr
        else:
            fwd_csr_ptrs = self._forward_graph[timestamp]
            self.fwd_row_offset_ptr = fwd_csr_ptrs.row_offset_ptr
            self.fwd_column_indices_ptr = fwd_csr_ptrs.column_indices_ptr
            self.fwd_eids_ptr = fwd_csr_ptrs.eids_ptr
            self.fwd_node_ids_ptr = fwd_csr_ptrs.node_ids_ptr

    def _update_graph_forward(self):
        """Updates the current base graph to the next timestamp"""
        if self.current_timestamp + 1 < self.total_timestamps:
            raise Exception(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()"
            )
        self._get_graph_csr_ptrs(self.current_timestamp + 1)

    def _init_reverse_graph(self):
        """Generates the reverse of the base graph"""
        self._get_graph_csr_ptrs(self.current_timestamp)

    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()"
            )
        self._get_graph_csr_ptrs(self.current_timestamp - 1)
