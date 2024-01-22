"""Base class for all STGraph temporal graph datasets."""

from __future__ import annotations

from rich.console import Console

from stgraph.dataset.STGraphDataset import STGraphDataset

console = Console()


class STGraphTemporalDataset(STGraphDataset):
    r"""Base class for temporal graph datasets.

    This class is a subclass of ``STGraphDataset`` and provides the base structure for
    handling temporal graph datasets.
    """

    def __init__(self: STGraphTemporalDataset) -> None:
        r"""Provide the base structure for handling temporal graph datasets."""
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self: STGraphTemporalDataset) -> dict:
        r"""Initialize graph meta data for a temporal dataset.

        The ``num_nodes``, ``num_edges``, ``total_timestamps`` keys are set to value 0
        """
        self.gdata["num_nodes"] = 0
        self.gdata["num_edges"] = 0
        self.gdata["total_timestamps"] = 0

        self._lags = 0
        self._cutoff_time = None
