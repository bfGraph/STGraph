from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from stgraph.graph.STGraphBase import STGraphBase


class StaticGraph(STGraphBase):
    def __init__(
        self: StaticGraph, edge_list: list, edge_weights: list, num_nodes: int
    ) -> None:
        super().__init__()
        self._num_nodes = num_nodes
        self._num_edges = len(set(edge_list))
        self._ndata = {}

    def get_num_nodes(self: StaticGraph) -> int:
        return self._num_nodes

    def get_num_edges(self: StaticGraph) -> int:
        return self._num_edges

    def get_ndata(self: StaticGraph, field: str) -> Any:
        if field in self._ndata:
            return self._ndata[field]
        else:
            return None

    def set_ndata(self: StaticGraph, field: str, val: Any) -> None:
        self._ndata[field] = val

    @abstractmethod
    def graph_type(self: StaticGraph) -> str:
        pass

    @abstractmethod
    def in_degrees(self: StaticGraph) -> Any:
        pass

    @abstractmethod
    def out_degrees(self: StaticGraph) -> Any:
        pass

    @abstractmethod
    def weighted_in_degrees(self: StaticGraph) -> Any:
        pass
