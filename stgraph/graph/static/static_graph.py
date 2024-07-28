"""Represent Static graphs in STGraph."""

from __future__ import annotations

import copy

import numpy as np
from rich.console import Console

from stgraph.graph.static.csr import CSR
from stgraph.graph.stgraph_base import STGraphBase

console = Console()


class StaticGraph(STGraphBase):
    r"""Represent Static graphs in STGraph.

    This abstract class outlines the interface for defining a static graphs
    used in STGraph. As of now the static graph is implemented using the
    Compressed Sparse Row (CSR) format.

    Example:
    -------
    .. code-block:: python

        from stgraph.graph import StaticGraph
        from stgraph.dataset import HungaryCPDataLoader

        hungary = HungaryCPDataLoader()

        graph = StaticGraph(
            edge_list = hungary.get_edges(),
            edge_weights = hungary.get_edge_weights(),
            num_nodes = hungary.gdata["num_nodes"]
        )

    """

    def __init__(
        self: StaticGraph,
        edge_list: list,
        edge_weights: list,
        num_nodes: int,
    ) -> None:
        r"""Represent Static graphs in STGraph."""
        super().__init__()
        self._num_nodes = num_nodes
        self._num_edges = len(set(edge_list))

        self._prepare_edge_lst_fwd(edge_list)
        self._forward_graph = CSR(
            self.fwd_edge_list,
            edge_weights,
            self._num_nodes,
            is_edge_reverse=True,
        )

        self._prepare_edge_lst_bwd(self.fwd_edge_list)
        self._backward_graph = CSR(self.bwd_edge_list, edge_weights, self._num_nodes)

        self._get_graph_csr_ptrs()

    # TODO-DOCS:
    def _prepare_edge_lst_fwd(self: STGraphBase, edge_list: list) -> None:
        edge_list_for_t = edge_list
        edge_list_for_t.sort(key=lambda x: (x[1], x[0]))
        edge_list_for_t = [
            (edge_list_for_t[j][0], edge_list_for_t[j][1], j)
            for j in range(len(edge_list_for_t))
        ]
        self.fwd_edge_list = edge_list_for_t

    # TODO-DOCS @nithin:
    def _prepare_edge_lst_bwd(self: STGraphBase, edge_list: list) -> None:
        edge_list_for_t = copy.deepcopy(edge_list)
        edge_list_for_t.sort()
        self.bwd_edge_list = edge_list_for_t

    # TODO-DOCS @nithin:
    def _get_graph_csr_ptrs(self: STGraphBase) -> None:
        self.fwd_row_offset_ptr = self._forward_graph.row_offset_ptr
        self.fwd_column_indices_ptr = self._forward_graph.column_indices_ptr
        self.fwd_eids_ptr = self._forward_graph.eids_ptr
        self.fwd_node_ids_ptr = self._forward_graph.node_ids_ptr

        self.bwd_row_offset_ptr = self._backward_graph.row_offset_ptr
        self.bwd_column_indices_ptr = self._backward_graph.column_indices_ptr
        self.bwd_eids_ptr = self._backward_graph.eids_ptr
        self.bwd_node_ids_ptr = self._backward_graph.node_ids_ptr

    def get_num_nodes(self: STGraphBase) -> int:
        r"""Return the number of nodes in the static graph."""
        return self._num_nodes

    def get_num_edges(self: STGraphBase) -> int:
        r"""Return the number of edges in the static graph."""
        return self._num_edges

    def get_ndata(self: STGraphBase, field: any) -> any:
        r"""Return the graph metadata."""
        if field in self._ndata:
            return self._ndata[field]

        return None

    def set_ndata(self: STGraphBase, field: str, val: any) -> None:
        r"""Set the graph metadata."""
        self._ndata[field] = val

    def graph_type(self: STGraphBase) -> str:
        r"""Return the graph type."""
        return "csr_unsorted"

    def in_degrees(self: STGraphBase) -> np.ndarray:
        r"""Return the graph inwards node degree array."""
        return np.array(self._forward_graph.out_degrees, dtype="int32")

    def out_degrees(self: STGraphBase) -> np.ndarray:
        r"""Return the graph outwards node degree array."""
        return np.array(self._forward_graph.in_degrees, dtype="int32")

    # TODO-DOCS @nithin:
    def weighted_in_degrees(self: STGraphBase) -> np.ndarray:
        r"""weighted_in_degrees."""
        return np.array(self._forward_graph.weighted_out_degrees, dtype="int32")
