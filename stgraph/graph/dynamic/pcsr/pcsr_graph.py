"""Represent Dynamic Graphs using PCSR in STGraph."""

from __future__ import annotations

import copy

import numpy as np

from stgraph.graph.dynamic.dynamic_graph import DynamicGraph
from stgraph.graph.dynamic.pcsr.pcsr import PCSR


class PCSRGraph(DynamicGraph):
    r"""Represent Dynamic Graphs using PCSR in STGraph.

    TODO: Add a paragraph explaining about PCSR in brief.

    Example:
    --------
    .. code-block:: python

        from stgraph.graph import PCSRGraph
        from stgraph.dataset import EnglandCovidDataLoader

        eng_covid = EnglandCovidDataLoader()

        G = PCSRGraph(
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

    def __init__(self: PCSRGraph, edge_list: list, max_num_nodes: int) -> None:
        r"""Represent Dynamic Graphs using PCSR in STGraph."""
        super().__init__(edge_list, max_num_nodes)

        # Get the maximum number of edges
        self._get_max_num_edges()

        self._forward_graph = PCSR(self.max_num_nodes, self.max_num_edges)
        self._forward_graph.edge_update_list(
            self.graph_updates["0"]["add"],
            is_reverse_edge=True,
        )
        self._forward_graph.label_edges()
        self._forward_graph.build_csr()
        self._get_graph_csr_ptrs()

        self.graph_cache = {}
        self.graph_cache["base"] = copy.deepcopy(self._forward_graph)

    def _get_max_num_edges(self: PCSRGraph) -> None:
        r"""TODO:."""
        updates = self.graph_updates
        edge_set = set()
        for i in range(len(updates)):
            for j in range(len(updates[str(i)]["add"])):
                edge_set.add(updates[str(i)]["add"][j])
        self.max_num_edges = len(edge_set)

    def graph_type(self: PCSRGraph) -> str:
        r"""Return the graph type."""
        return "pcsr"

    def _cache_graph(self: PCSRGraph) -> None:
        r"""TODO:."""
        self.graph_cache[str(self.current_timestamp)] = copy.deepcopy(
            self._forward_graph,
        )

    def _get_cached_graph(self: PCSRGraph, timestamp: int | str) -> bool:
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

    def in_degrees(self: PCSRGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(self._forward_graph.out_degrees, dtype="int32")

    def out_degrees(self: PCSRGraph) -> np.ndarray:
        r"""TODO:."""
        return np.array(self._forward_graph.in_degrees, dtype="int32")

    def _get_graph_csr_ptrs(self: PCSRGraph) -> None:
        r"""TODO:."""
        csr_ptrs = self._forward_graph.get_csr_ptrs()
        if self._is_backprop_state:
            self.bwd_row_offset_ptr = csr_ptrs[0]
            self.bwd_column_indices_ptr = csr_ptrs[1]
            self.bwd_eids_ptr = csr_ptrs[2]
            self.bwd_node_ids_ptr = csr_ptrs[3]
        else:
            self.fwd_row_offset_ptr = csr_ptrs[0]
            self.fwd_column_indices_ptr = csr_ptrs[1]
            self.fwd_eids_ptr = csr_ptrs[2]
            self.fwd_node_ids_ptr = csr_ptrs[3]

    def _update_graph_forward(self: PCSRGraph) -> None:
        r"""Update the current base graph to the next timestamp."""
        if str(self.current_timestamp + 1) not in self.graph_updates:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()",
            )

        graph_additions = self.graph_updates[str(self.current_timestamp + 1)]["add"]
        graph_deletions = self.graph_updates[str(self.current_timestamp + 1)]["delete"]

        self._forward_graph.edge_update_list(graph_additions, is_reverse_edge=True)
        self._forward_graph.edge_update_list(
            graph_deletions,
            is_delete=True,
            is_reverse_edge=True,
        )
        self._forward_graph.label_edges()
        move_to_gpu_time = self._forward_graph.build_csr()
        self.move_to_gpu_time += move_to_gpu_time
        self._get_graph_csr_ptrs()

    def _init_reverse_graph(self: PCSRGraph) -> None:
        """Generate the reverse of the base graph."""
        move_to_gpu_time = self._forward_graph.build_reverse_csr()
        self.move_to_gpu_time += move_to_gpu_time
        self._get_graph_csr_ptrs()

    def _update_graph_backward(self: PCSRGraph) -> None:
        r"""TODO:."""
        if self.current_timestamp < 0:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()",
            )

        graph_additions = self.graph_updates[str(self.current_timestamp)]["delete"]
        graph_deletions = self.graph_updates[str(self.current_timestamp)]["add"]

        self._forward_graph.edge_update_list(graph_additions, is_reverse_edge=True)
        self._forward_graph.edge_update_list(
            graph_deletions,
            is_delete=True,
            is_reverse_edge=True,
        )
        self._forward_graph.label_edges()
        move_to_gpu_time = self._forward_graph.build_reverse_csr()
        self.move_to_gpu_time += move_to_gpu_time
        self._get_graph_csr_ptrs()
