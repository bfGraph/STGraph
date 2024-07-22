"""Graph Convolutional Network Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.nn.functional import relu

from stgraph.nn.pytorch.static.gcn_conv import GCNConv

if TYPE_CHECKING:
    from stgraph.graph import StaticGraph


class GCN(nn.Module):
    r"""Graph Convolutional Network Model.

    A multi-layer Graph Convolutional Network Model for node classification task.

    Parameters
    ----------
    graph : StaticGraph
        The input static graph the GCN model operates.
    in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units in a hidden layer.
    n_classes : int
        Number of output classes.
    n_hidden_layers : int
        Number of hidden layers.

    """

    def __init__(
        self: GCN,
        graph: StaticGraph,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_hidden_layers: int,
    ) -> None:
        r"""Graph Convolutional Network Model."""
        super().__init__()

        self._graph = graph
        self._layers = nn.ModuleList()

        # input layer
        self._layers.append(GCNConv(in_feats, n_hidden, relu, bias=True))

        # hidden layers
        for _ in range(n_hidden_layers):
            self._layers.append(GCNConv(n_hidden, n_hidden, relu, bias=True))

        # output layer
        self._layers.append(GCNConv(n_hidden, n_classes, None, bias=True))

    def forward(self: GCN, features: Tensor) -> Tensor:
        r"""Forward pass of the GCN model.

        Parameters
        ----------
        features : Tensor
            Input features for each node in the graph.

        Returns
        -------
        Tensor :
            The output features after applying all the GCN layers.

        """
        h = features
        for layer in self._layers:
            h = layer.forward(self._graph, h)
        return h
