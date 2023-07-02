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

        self._get_graph_csr_ptrs()

        self.graph_cache = {}
        self.graph_cache["base"] = copy.deepcopy(self._forward_graph)

    def graph_type(self):
        return "gpma"

    def _cache_graph(self):
        self.graph_cache[str(self.current_timestamp)] = copy.deepcopy(self._forward_graph)

    def _get_cached_graph(self, timestamp):
        if timestamp == "base":
            self._forward_graph = copy.deepcopy(self.graph_cache["base"])
            self._get_graph_csr_ptrs()
            return True
        else:
            if str(timestamp) in self.graph_cache:
                self._forward_graph = self.graph_cache[str(timestamp)]
                del self.graph_cache[str(timestamp)]
                self._get_graph_csr_ptrs()
                return True
            else:
                return False
    
    def in_degrees(self):
        return np.array(get_out_degrees(self._forward_graph), dtype="int32")
    
    def out_degrees(self):
        return np.array(get_in_degrees(self._forward_graph), dtype="int32")

    def _get_graph_csr_ptrs(self):

        forward_csr_ptrs = get_csr_ptrs(self._forward_graph)
        self.fwd_row_offset_ptr = forward_csr_ptrs[0]
        self.fwd_column_indices_ptr = forward_csr_ptrs[1]
        self.fwd_eids_ptr = forward_csr_ptrs[2]
        self.fwd_node_ids_ptr = forward_csr_ptrs[3]

        if self._is_backprop_state:
            backward_csr_ptrs = get_csr_ptrs(self._forward_graph, is_backward=True)
            self.bwd_row_offset_ptr = backward_csr_ptrs[0]
            self.bwd_column_indices_ptr = backward_csr_ptrs[1]
            self.bwd_eids_ptr = backward_csr_ptrs[2]
            self.bwd_node_ids_ptr = backward_csr_ptrs[3]

    def _update_graph_forward(self):
        # if we went through the entire time-stamps
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_forward()"
            )

        edge_update_t(self._forward_graph, self.current_timestamp + 1)
        label_edges(self._forward_graph)
        self._get_graph_csr_ptrs()

    def _init_reverse_graph(self):
        """Generates the reverse of the base graph"""
        free_backward_csr(self._forward_graph)
        build_backward_csr(self._forward_graph)
        self._get_graph_csr_ptrs()

    def _update_graph_backward(self):
        if self.current_timestamp < 0:
            raise Exception(
                "⏰ Invalid timestamp during SeastarGraph.update_graph_backward()"
            )

        # Freeing resources from previous CSR
        free_backward_csr(self._forward_graph)
        edge_update_t(
            self._forward_graph, self.current_timestamp, revert_update=True
        )
        label_edges(self._forward_graph)
        build_backward_csr(self._forward_graph)
        self._get_graph_csr_ptrs()
