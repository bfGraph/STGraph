import copy
import time

import numpy as np
from rich import inspect

from seastar.graph.dynamic.DynamicGraph import DynamicGraph
from seastar.graph.dynamic.gpma.gpma import (
    GPMA,
    init_gpma,
    init_graph_updates,
    edge_update_t,
    label_edges,
    build_backward_csr,
    free_backward_csr,
    get_csr_ptrs,
    get_out_degrees,
    get_in_degrees
)


class GPMAGraph(DynamicGraph):
    def __init__(self, edge_list, max_num_nodes):
        super().__init__(edge_list, max_num_nodes)

        # forward graph for GPMA
        self._forward_graph = GPMA()
        init_gpma(self._forward_graph, self.max_num_nodes)
        init_graph_updates(self._forward_graph, self.graph_updates, reverse_edges=True)

        # base forward graph at t=0
        edge_update_t(self._forward_graph, 0)
        label_edges(self._forward_graph)

        # Cacheing node degrees
        self._in_degrees_cache = {}
        self._out_degrees_cache = {}

        # for benchmarking purposes
        self._update_count = 0
        self._total_update_time = 0
        self._gpu_move_time = 0

        self._get_graph_csr_ptrs()

    def graph_type(self):
        return "gpma"
    
    def in_degrees(self):
        if self.current_timestamp not in self._in_degrees_cache:
            self._in_degrees_cache[self.current_timestamp] = np.array(get_out_degrees(self._forward_graph), dtype="int32")
        
        return self._in_degrees_cache[self.current_timestamp]
    
    def out_degrees(self):
        if self.current_timestamp not in self._out_degrees_cache:
            self._out_degrees_cache[self.current_timestamp] = np.array(get_in_degrees(self._forward_graph), dtype="int32")
        
        return self._out_degrees_cache[self.current_timestamp]

    def _get_graph_csr_ptrs(self):

        forward_csr_ptrs = get_csr_ptrs(self._forward_graph)
        self.fwd_row_offset_ptr = forward_csr_ptrs[0]
        self.fwd_column_indices_ptr = forward_csr_ptrs[1]
        self.fwd_eids_ptr = forward_csr_ptrs[2]

        if self._is_backprop_state:
          backward_csr_ptrs = get_csr_ptrs(self._forward_graph, is_backward=True)
          self.bwd_row_offset_ptr = backward_csr_ptrs[0]
          self.bwd_column_indices_ptr = backward_csr_ptrs[1]
          self.bwd_eids_ptr = backward_csr_ptrs[2]

    def _update_graph_forward(self):
        # if we went through the entire time-stamps
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_forward()"
            )

        self._update_count += len(
            self.graph_updates[str(self.current_timestamp + 1)]["add"]
        )
        self._update_count += len(
            self.graph_updates[str(self.current_timestamp + 1)]["delete"]
        )

        update_time_0 = time.time()

        edge_update_t(self._forward_graph, self.current_timestamp + 1)
        label_edges(self._forward_graph)

        update_time_1 = time.time()
        self._total_update_time += update_time_1 - update_time_0

        self._get_graph_csr_ptrs()

    def _init_reverse_graph(self):
        """Generates the reverse of the base graph"""
        build_backward_csr(self._forward_graph)
        self._get_graph_csr_ptrs()

    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_backward()"
            )

        self._update_count += len(
            self.graph_updates[str(self.current_timestamp)]["delete"]
        )
        self._update_count += len(
            self.graph_updates[str(self.current_timestamp)]["add"]
        )

        update_time_0 = time.time()

        # Freeing resources from previous CSR
        free_backward_csr(self._forward_graph)

        edge_update_t(
            self._forward_graph, self.current_timestamp, revert_update=True
        )
        label_edges(self._forward_graph)
        build_backward_csr(self._forward_graph)

        update_time_1 = time.time()
        self._total_update_time += update_time_1 - update_time_0

        self._get_graph_csr_ptrs()
