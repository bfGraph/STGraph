"""Graph Convolutional Network Layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from stgraph.compiler.node import CentralNode
    from stgraph.graph import StaticGraph

import torch
from torch import Tensor, nn

from stgraph.compiler import STGraph
from stgraph.compiler.backend.pytorch.torch_callback import STGraphBackendTorch
from stgraph.utils.constants import SizeConstants


class GCNConv(nn.Module):
    r"""Graph Convolutional Network Layer.

    Vertex-centric implementation for Graph Convolutional Network (GCN)
    layer as described in `Semi-supervised Classification with Graph
    Convolutional Networks <https://arxiv.org/abs/1609.02907>`_.

    A multi-layer GCN model has the following layer-wise propagation rule

    .. math::

        H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)

    - :math:`H^{(l)}`: Matrix of activations in the :math:`l`-th layer; :math:`H^{(0)} = X` is the input feature matrix.
    - :math:`\sigma`: Activation function (e.g., ReLU).
    - :math:`\tilde{A} = A + I_N`: Adjacency matrix of the graph with added self-connections.
    - :math:`I_N`: Identity matrix.
    - :math:`\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}`: Degree matrix of :math:`\tilde{A}`.
    - :math:`W^{(l)}`: Trainable weight matrix for the :math:`l`-th layer.

    **Vertex-Centric Formula**

    The vertex-centric implementation can be achieved by aggregating all the
    features of the neighbouring nodes of the central node

    .. math::

        h^{(l+1)} = \left( \sum_{\text{nb} \in \text{innbs}(v)} \text{nb}_{h^{(l)}} \cdot \text{nb}_{\text{norm}} \cdot \text{weight}_{\text{nb,v}} \right) \cdot v_{\text{norm}}

    - :math:`h^{(l)}`: Activations of central-node in the :math:`l`-th layer.
    - :math:`\text{innbs}(v)`: In-neighbours of central-node :math:`v`.
    - :math:`\text{weight}_{\text{nb,v}}`: Weight of edge from :math:`nb` to :math:`v`. In case no edge weights are present, it is set to 1
    - :math:`norm`: Node wise normalization factor, :math:`v_{\text{norm}} = \text{in_degrees(v)}^{-0.5}`.

    **Node Data**

    The following node data needs to be set using :class:`StaticGraph.set_ndata <stgraph.graph.static.static_graph.StaticGraph>` before calling
    the :func:`~stgraph.nn.pytorch.static.gcn_conv.GCNConv.forward` method.

    +---------------+--------------------------------+---------------------------------------------------------------------------------------------------+
    | Node Property | Description                    | Type                                                                                              |
    +===============+================================+===================================================================================================+
    | norm          | Node-wise normalization factor | A PyTorch Tensor of shape (num_nodes, 1), where dim=1 contains the node-wise normalization factor |
    +---------------+--------------------------------+---------------------------------------------------------------------------------------------------+


    Parameters
    ----------
    in_channels : int
        Size of input sample passed into the layer
    out_channels : int
        Size of output sample outputted by the layer
    activation : optional
        Non-linear activation function provided by `PyTorch <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`_
    bias : bool, optional
        If set to *True*, learnable bias parameters are added to the layer

    """

    def __init__(
        self: GCNConv,
        in_channels: int,
        out_channels: int,
        activation: Callable[..., torch.Tensor] | None = None,
        bias: bool = True,
    ) -> None:
        """Graph Convolutional Network Layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.activation = activation
        self.stgraph = STGraph(STGraphBackendTorch())
        self.reset_parameters()

    def reset_parameters(self: GCNConv) -> None:
        r"""Reset the learnable weight and bias parameters.

        The weight parameter is initialized using a Xavier Uniform distribution.
        The bias parameter is initialized by setting all values to zero.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self: GCNConv,
        graph: StaticGraph,
        h: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        r"""Execute a single forward pass for the GCN layer.

        Runs a single forward pass using the vertex-centric implementation of the GCN layer.

        Parameters
        ----------
        graph : StaticGraph
            A StaticGraph graph object
        h : Tensor
            Input for the GCN forward pass
        edge_weight : Tensor, optional
            Edge weights for each edge in the graph

        Returns
        -------
        Tensor
            The output after executing the GCN forward pass

        Raises
        ------
        KeyError
            If ``norm`` n_data is not present for the graph
        ValueError
            If ``norm`` n_data passed is not of the shape (num_nodes, 1)

        Example
        -------

        Example usage::

            # Defining a method to run forward pass with multiple GCN layers

            def forward(input: Tensor, layers: List[GCNConv], graph: StaticGraph):
                h = input
                for layer in layers:
                    h = layer.forward(graph, h)
                return h

        """
        if graph.get_ndata("norm") is None:
            raise KeyError("StaticGraph passed to GCNConv forward pass does not contain 'norm' node data")
        if (len(graph.get_ndata("norm").shape) != SizeConstants.NODE_NORM_SIZE.value or
                graph.get_ndata("norm").shape[1] != 1 or
                graph.get_ndata("norm").shape[0] != graph.get_num_nodes()):
            raise ValueError("Node data 'norm' passed to GCNConv should be of shape (num_nodes, 1)")

        h = torch.mm(h, self.weight)

        if edge_weight is None:

            @self.stgraph.compile(gnn_module=self)
            def nb_compute(v: CentralNode) -> Tensor:
                return sum([nb.h * nb.norm for nb in v.innbs]) * v.norm

            h = nb_compute(g=graph, n_feats={"norm": graph.get_ndata("norm"), "h": h})
        else:

            @self.stgraph.compile(gnn_module=self)
            def nb_compute(v: CentralNode) -> Tensor:
                return sum(
                    [
                        nb_edge.src.norm * nb_edge.src.h * nb_edge.edge_weight
                        for nb_edge in v.inedges
                    ],
                ) * v.norm

            h = nb_compute(
                g=graph,
                n_feats={"norm": graph.get_ndata("norm"), "h": h},
                e_feats={"edge_weight": edge_weight},
            )

        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h
