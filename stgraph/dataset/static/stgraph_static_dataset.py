"""Base class for all STGraph static graph datasets"""

from rich.console import Console

from stgraph.dataset.STGraphDataset import STGraphDataset


console = Console()


class STGraphStaticDataset(STGraphDataset):
    r"""Base class for static graph datasets

    This class is a subclass of ``STGraphDataset`` and provides the base structure for
    handling static graph datasets.
    """

    def __init__(self) -> None:
        super().__init__()

        self._init_graph_data()

    def _init_graph_data(self) -> dict:
        r"""Initialize graph meta data for a static dataset.

        The ``num_nodes`` and ``num_edges`` keys are set to value 0
        """
        self.gdata["num_nodes"] = 0
        self.gdata["num_edges"] = 0
