"""Represent Dynamic Graphs using GPMA in STGraph."""

from __future__ import annotations

import copy

import numpy as np

from stgraph.graph.dynamic.dynamic_graph import DynamicGraph
from stgraph.graph.dynamic.gpma.gpma import (
    GPMA,
    build_backward_csr,
    edge_update_t,
    free_backward_csr,
    get_csr_ptrs,
    get_in_degrees,
    get_out_degrees,
    init_gpma,
    init_graph_updates,
    label_edges,
)


class GPMAGraph(DynamicGraph):
    r"""Represent Dynamic Graphs using GPMA in STGraph.

    TODO: Add a paragraph explaining about GPMA in brief.

    Example:
    --------
    .. code-block:: python

        from stgraph.graph import GPMAGraph
        from stgraph.dataset import EnglandCovidDataLoader

        eng_covid = EnglandCovidDataLoader()

        G = GPMAGraph(
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

    def __init__(self: GPMAGraph, edge_list: list, max_num_nodes: int) -> None:
        r"""Represent Dynamic Graphs using GPMA in STGraph."""
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

    def graph_type(self: GPMAGraph) -> str:
        r"""Return the graph type."""
        return "gpma"

    def _cache_graph(self: GPMAGraph) -> None:
        r"""TODO:."""
        self.graph_cache[str(self.current_timestamp)] = copy.deepcopy(
            self._forward_graph,
        )

    def _get_cached_graph(self: GPMAGraph, timestamp: int | str) -> bool:
        r"""TODO:."""
        if timestamp == "base":
            self._forward_graph = copy.deepcopy(self.graph_cache["base"])
            self._get_graph_csr_ptrs()
            return True

        if str(timestamp) in self.graph_cache:
            self._forward_graph = self.graph_cache[str(timestamp)]
            del self.graph_cache[str(timestamp)]
            self._get_graph_csr_ptrs()
            return True

        return False

    def in_degrees(self: GPMAGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(get_out_degrees(self._forward_graph), dtype="int32")

    def out_degrees(self: GPMAGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(get_in_degrees(self._forward_graph), dtype="int32")

    def _get_graph_csr_ptrs(self: GPMAGraph) -> None:
        r"""TODO:."""
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

    def _update_graph_forward(self: GPMAGraph) -> None:
        r"""TODO:."""
        # if we went through the entire time-stamps
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()",
            )

        edge_update_t(self._forward_graph, self.current_timestamp + 1)
        label_edges(self._forward_graph)
        self._get_graph_csr_ptrs()

    def _init_reverse_graph(self: GPMAGraph) -> None:
        r"""Generate the reverse of the base graph."""
        free_backward_csr(self._forward_graph)
        build_backward_csr(self._forward_graph)
        self._get_graph_csr_ptrs()

    def _update_graph_backward(self: GPMAGraph) -> None:
        r"""TODO:."""
        if self.current_timestamp < 0:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()",
            )

        # Freeing resources from previous CSR
        free_backward_csr(self._forward_graph)
        edge_update_t(self._forward_graph, self.current_timestamp, revert_update=True)
        label_edges(self._forward_graph)
        build_backward_csr(self._forward_graph)
        self._get_graph_csr_ptrs()
