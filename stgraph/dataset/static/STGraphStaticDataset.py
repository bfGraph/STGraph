"""Base class for all STGraph static graph datasets"""


from abc import ABC, abstractmethod
from rich.console import Console

from stgraph.dataset.STGraphDataset import STGraphDataset


console = Console()


class STGraphStaticDataset(STGraphDataset):
    def __init__(self) -> None:
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self) -> dict:
        self.gdata["num_nodes"] = 0
        self.gdata["num_edges"] = 0
