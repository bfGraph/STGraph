"""Represent Dynamic Graphs using CSR in STGraph."""

from __future__ import annotations

import copy

import numpy as np

from stgraph.graph.dynamic.dynamic_graph import DynamicGraph
from stgraph.graph.static.csr import CSR


class NaiveGraph(DynamicGraph):
    r"""Represent Dynamic Graphs using CSR in STGraph.

    TODO: Add a paragraph explaining about GPMA in brief.

    Example:
    --------
    .. code-block:: python

        from stgraph.graph import NaiveGraph
        from stgraph.dataset import EnglandCovidDataLoader

        eng_covid = EnglandCovidDataLoader()

        G = NaiveGraph(
            edge_list = eng_covid.get_edges(),
            max_num_nodes = max(eng_covid.gdata["num_nodes"]),
        )

    Parameters
    ----------
    edge_list : list
        Edge list of the graph across all timestamps
    max_num_nodes : int
        Maximum number of nodes present in the graph across all timestamps

    Attributes
    ----------
    TODO:.

    """

    def __init__(self: NaiveGraph, edge_list: list, max_num_nodes: int) -> None:
        r"""Represent Dynamic Graphs using CSR in STGraph."""
        super().__init__(edge_list, max_num_nodes)

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

    def _prepare_edge_lst_fwd(self: NaiveGraph, edge_list: list) -> None:
        r"""TODO:."""
        self.fwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = edge_list[i]
            edge_list_for_t.sort(key=lambda x: (x[1], x[0]))
            edge_list_for_t = [
                (edge_list_for_t[j][0], edge_list_for_t[j][1], j)
                for j in range(len(edge_list_for_t))
            ]
            self.fwd_edge_list.append(edge_list_for_t)

    def _prepare_edge_lst_bwd(self: NaiveGraph, edge_list: list) -> None:
        r"""TODO:."""
        self.bwd_edge_list = []
        for i in range(len(edge_list)):
            edge_list_for_t = copy.deepcopy(edge_list[i])
            edge_list_for_t.sort()
            self.bwd_edge_list.append(edge_list_for_t)

    def graph_type(self: NaiveGraph) -> str:
        r"""Return the graph type."""
        return "csr"

    def _cache_graph(self: NaiveGraph) -> None:
        pass

    def _get_cached_graph(self: NaiveGraph) -> None:
        return False

    def in_degrees(self: NaiveGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(
            self._forward_graph[self.current_timestamp].out_degrees, dtype="int32",
        )

    def out_degrees(self: NaiveGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(
            self._forward_graph[self.current_timestamp].in_degrees, dtype="int32",
        )

    def _get_graph_csr_ptrs(self: NaiveGraph, timestamp: int) -> None:
        r"""TODO:."""
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

    def _update_graph_forward(self: NaiveGraph) -> None:
        """Update the current base graph to the next timestamp."""
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()",
            )
        self._get_graph_csr_ptrs(self.current_timestamp + 1)

    def _init_reverse_graph(self: NaiveGraph) -> None:
        """Generate the reverse of the base graph."""
        self._get_graph_csr_ptrs(self.current_timestamp)

    def _update_graph_backward(self: NaiveGraph) -> None:
        r"""TODO:."""
        if self.current_timestamp < 0:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()",
            )
        self._get_graph_csr_ptrs(self.current_timestamp - 1)
