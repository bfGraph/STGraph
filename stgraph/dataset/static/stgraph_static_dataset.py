"""Base class for all STGraph static graph datasets."""

from __future__ import annotations

from stgraph.dataset.STGraphDataset import STGraphDataset


class STGraphStaticDataset(STGraphDataset):
    r"""Base class for static graph datasets."""

    def __init__(self: STGraphStaticDataset) -> None:
        r"""Provide the base structure for handling static graph datasets."""
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self: STGraphStaticDataset) -> dict:
        r"""Initialize graph meta data for a static dataset.

        The ``num_nodes`` and ``num_edges`` keys are set to value 0
        """
        self.gdata["num_nodes"] = 0
        self.gdata["num_edges"] = 0
