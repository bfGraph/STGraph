"""Base class for all STGraph dynamic graph datasets"""

from stgraph.dataset.STGraphDataset import STGraphDataset


class STGraphDynamicDataset(STGraphDataset):
    r"""Base class for dynamic graph datasets

    This class is a subclass of ``STGraphDataset`` and provides the base structure for
    handling dynamic graph datasets."""

    def __init__(self) -> None:
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self) -> dict:
        r"""Initialize graph meta data for a dynamic dataset.

        The ``num_nodes``, ``num_edges``, ``total_timestamps``, ``max_num_nodes`` and ``max_num_edges``
        keys are set to value 0
        """
        self.gdata["num_nodes"] = {}
        self.gdata["num_edges"] = {}
        self.gdata["total_timestamps"] = 0
        self.gdata["max_num_nodes"] = 0
        self.gdata["max_num_edges"] = 0

        self._lags = 0
        self._cutoff_time = None
