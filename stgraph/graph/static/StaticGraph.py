from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from stgraph.graph.STGraphBase import STGraphBase


class StaticGraph(STGraphBase):
    def __init__(self: StaticGraph) -> None:
        super().__init__()
        self.num_nodes = 0
        self.num_edges = 0
        self.ndata = {}

    def get_num_nodes(self: StaticGraph) -> int:
        return self.num_nodes

    def get_num_edges(self: StaticGraph) -> int:
        return self.num_edges

    def get_ndata(self: StaticGraph, field: str) -> Any:
        if field in self.ndata:
            return self.ndata[field]
        else:
            return None

    def set_ndata(self: StaticGraph, field: str, val: Any) -> None:
        self.ndata[field] = val

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
