from __future__ import annotations

import copy
import numpy as np

from stgraph.graph.static.StaticGraph import StaticGraph
from stgraph.graph.static.csr.csr import CSR


class CSRGraph(StaticGraph):
    def __init__(
        self: CSRGraph, edge_list: list, edge_weights: list, num_nodes: int
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = len(set(edge_list))

        self._fwd_edge_list = None
        self._bwd_edge_list = None
        self._fwd_row_offset_ptr = None
        self._fwd_column_indices_ptr = None
        self._fwd_eids_ptr = None
        self._fwd_node_ids_ptr = None
        self._bwd_row_offset_ptr = None
        self._bwd_column_indices_ptr = None
        self._bwd_eids_ptr = None
        self._bwd_node_ids_ptr = None

        self._prepare_edge_lst_fwd(edge_list)
        self._forward_graph = CSR(
            self.fwd_edge_list, edge_weights, self._num_nodes, is_edge_reverse=True
        )

        self._prepare_edge_lst_bwd(self.fwd_edge_list)
        self._backward_graph = CSR(self.bwd_edge_list, edge_weights, self._num_nodes)

        self._set_graph_csr_ptrs()

    def _prepare_edge_lst_fwd(self: CSRGraph, edge_list: list) -> None:
        edge_list_for_t = edge_list
        edge_list_for_t.sort(key=lambda x: (x[1], x[0]))
        edge_list_for_t = [
            (edge_list_for_t[j][0], edge_list_for_t[j][1], j)
            for j in range(len(edge_list_for_t))
        ]
        self._fwd_edge_list = edge_list_for_t

    def _prepare_edge_lst_bwd(self: CSRGraph, edge_list: list) -> None:
        edge_list_for_t = copy.deepcopy(edge_list)
        edge_list_for_t.sort()
        self._bwd_edge_list = edge_list_for_t

    def _set_graph_csr_ptrs(self: CSRGraph) -> None:
        self._fwd_row_offset_ptr = self._forward_graph.row_offset_ptr
        self._fwd_column_indices_ptr = self._forward_graph.column_indices_ptr
        self._fwd_eids_ptr = self._forward_graph.eids_ptr
        self._fwd_node_ids_ptr = self._forward_graph.node_ids_ptr

        self._bwd_row_offset_ptr = self._backward_graph.row_offset_ptr
        self._bwd_column_indices_ptr = self._backward_graph.column_indices_ptr
        self._bwd_eids_ptr = self._backward_graph.eids_ptr
        self._bwd_node_ids_ptr = self._backward_graph.node_ids_ptr

    def graph_type(self: CSRGraph) -> str:
        return "csr_unsorted"

    def in_degrees(self: CSRGraph) -> np.ndarray:
        return np.array(self._forward_graph.out_degrees, dtype="int32")

    def out_degrees(self: CSRGraph) -> np.ndarray:
        return np.array(self._forward_graph.in_degrees, dtype="int32")

    def weighted_in_degrees(self: CSRGraph) -> np.ndarray:
        return np.array(self._forward_graph.weighted_out_degrees, dtype="int32")
