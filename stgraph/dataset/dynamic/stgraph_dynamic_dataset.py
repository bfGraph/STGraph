"""Base class for all STGraph dynamic graph datasets."""

from __future__ import annotations

from stgraph.dataset.stgraph_dataset import STGraphDataset


class STGraphDynamicDataset(STGraphDataset):
    r"""Base class for dynamic graph datasets."""

    def __init__(self: STGraphDynamicDataset) -> None:
        r"""Provide the base structure for handling dynamic graph datasets."""
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self: STGraphDynamicDataset) -> dict:
        r"""Initialize graph meta data for a dynamic dataset.

        The ``num_nodes``, ``num_edges``, ``total_timestamps`` keys are set to value 0
        """
        self.gdata["num_nodes"] = {}
        self.gdata["num_edges"] = {}
        self.gdata["total_timestamps"] = 0

        self._lags = 0
        self._cutoff_time = None
