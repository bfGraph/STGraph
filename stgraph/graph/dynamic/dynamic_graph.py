"""Represent Dynamic Graphs in STGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import time
from abc import abstractmethod

from stgraph.graph.stgraph_base import STGraphBase


class DynamicGraph(STGraphBase):
    r"""Represent Dynamic Graphs in STGraph.

    This abstract class outlines the interface for defining a dynamic graph
    used in STGraph. As of now the dynamic graph is implemented using the
    following graph representation format:

    1. Compressed Sparse Row (CSR)
    2. Packed Compressed Sparse Row (PCSR)
    3. GPMA

    Please note that this documentation is still work in progress.

    """

    def __init__(
        self: DynamicGraph,
        edge_list: list,
        max_num_nodes: int,
    ) -> None:
        r"""Represent Dynamic Graphs in STGraph."""
        super().__init__()
        self.graph_updates = {}
        self.max_num_nodes = max_num_nodes
        self.graph_attr = {
            str(t): (self.max_num_nodes, len(set(edge_list[t])))
            for t in range(len(edge_list))
        }

        # Indicates whether the graph is currently undergoing backprop
        self._is_backprop_state = False
        self.current_timestamp = 0

        # Measuring time for operations
        self.get_fwd_graph_time = 0
        self.get_bwd_graph_time = 0
        self.move_to_gpu_time = 0

        self._preprocess_graph_structure(edge_list)

    def _preprocess_graph_structure(self: DynamicGraph, edge_list: list) -> None:
        r"""TODO:."""
        edge_dict = {}
        for i in range(len(edge_list)):
            edge_set = set()
            for j in range(len(edge_list[i])):
                edge_set.add((edge_list[i][j][0], edge_list[i][j][1]))
            edge_dict[str(i)] = edge_set

        self.graph_updates = {}

        # Presorting additions and deletions (is a manadatory step for GPMA)
        additions = list(edge_dict["0"])
        additions.sort(key=lambda x: (x[1], x[0]))
        self.graph_updates["0"] = {"add": additions, "delete": []}
        for i in range(1, len(edge_list)):
            additions = list(edge_dict[str(i)].difference(edge_dict[str(i - 1)]))
            additions.sort(key=lambda x: (x[1], x[0]))
            deletions = list(edge_dict[str(i - 1)].difference(edge_dict[str(i)]))
            deletions.sort(key=lambda x: (x[1], x[0]))
            self.graph_updates[str(i)] = {
                "add": additions,
                "delete": deletions,
            }

    def reset_graph(self: DynamicGraph) -> None:
        r"""TODO:."""
        self._get_cached_graph("base")
        self.current_timestamp = 0

        self.get_fwd_graph_time = 0
        self.get_bwd_graph_time = 0
        self.move_to_gpu_time = 0

    def get_graph(self: DynamicGraph, timestamp: int) -> None:
        r"""TODO:."""
        t0 = time.time()

        self._is_backprop_state = False

        if timestamp < self.current_timestamp:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_forward()",
            )

        if self._get_cached_graph(timestamp - 1):
            self.current_timestamp = timestamp - 1

        while self.current_timestamp < timestamp:
            self._update_graph_forward()
            self.current_timestamp += 1

        self.get_fwd_graph_time += time.time() - t0

    def get_backward_graph(self: DynamicGraph, timestamp: int) -> None:
        r"""TODO:."""
        t0 = time.time()

        if not self._is_backprop_state:
            self._cache_graph()
            self._is_backprop_state = True
            self._init_reverse_graph()

        if timestamp > self.current_timestamp:
            raise RuntimeError(
                "⏰ Invalid timestamp during STGraphBase.update_graph_backward()",
            )

        while self.current_timestamp > timestamp:
            self._update_graph_backward()
            self.current_timestamp -= 1

        self.get_bwd_graph_time += time.time() - t0

    def get_num_nodes(self: DynamicGraph) -> int:
        r"""TODO:."""
        return self.graph_attr[str(self.current_timestamp)][0]

    def get_num_edges(self: DynamicGraph) -> int:
        r"""TODO:."""
        return self.graph_attr[str(self.current_timestamp)][1]

    def get_ndata(self: DynamicGraph, field: str) -> any:
        r"""TODO:."""
        if (
            str(self.current_timestamp) in self._ndata
            and field in self._ndata[str(self.current_timestamp)]
        ):
            return self._ndata[str(self.current_timestamp)][field]

        return None

    def set_ndata(self: DynamicGraph, field: str, val: any) -> None:
        r"""TODO:."""
        if str(self.current_timestamp) in self._ndata:
            self._ndata[str(self.current_timestamp)][field] = val
        else:
            self._ndata[str(self.current_timestamp)] = {field: val}

    @abstractmethod
    def in_degrees(self: DynamicGraph) -> np.ndarray:
        r"""TODO:."""
        pass

    @abstractmethod
    def out_degrees(self: DynamicGraph) -> np.ndarray:
        r"""TODO:."""
        pass

    @abstractmethod
    def _cache_graph(self: DynamicGraph) -> None:
        r"""TODO:."""
        pass

    @abstractmethod
    def _get_cached_graph(self: DynamicGraph, timestamp: str | int) -> bool:
        r"""TODO:."""
        pass

    @abstractmethod
    def _update_graph_forward(self: DynamicGraph) -> None:
        r"""TODO:."""
        pass

    @abstractmethod
    def _init_reverse_graph(self: DynamicGraph) -> None:
        r"""TODO:."""
        pass

    @abstractmethod
    def _update_graph_backward(self: DynamicGraph) -> None:
        r"""TODO:."""
        pass
