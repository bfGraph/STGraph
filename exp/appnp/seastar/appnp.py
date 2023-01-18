"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch as th
import dgl
import torch.nn as nn
from seastar import CtxManager

class EglAPPNPConv(nn.Module):
    r"""Approximate Personalized Propagation of Neural Predictions
    layer from paper `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`__.

    .. math::
        H^{0} & = X

        H^{t+1} & = (1-\alpha)\left(\hat{D}^{-1/2}
        \hat{A} \hat{D}^{-1/2} H^{t}\right) + \alpha H^{0}

    Parameters
    ----------
    k : int
        Number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        Dropout rate on edges that controls the
        messages received by each node. Default: ``0``.
    """
    def __init__(self,
                 k,
                 alpha,
                 edge_drop=0.):
        super(EglAPPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)
        self.cm = CtxManager(dgl.backend.run_egl)

    def forward(self, graph, feat):
        r"""Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        graph = graph.local_var()
        norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)
        feat_0 = feat
        dgl_context = dgl.utils.to_dgl_context(feat.device)
        graph = graph._graph.get_immutable_gidx(dgl_context)
        @self.cm.zoomIn(nspace=[self, th])
        def nb_compute(v):
            for _ in range(self._k):
                feat = sum([nb.feat * nb.norm for nb in v.innbs])
                feat = feat * v.norm
                feat = (1 - self._alpha) * feat + self._alpha * v.feat_0
                v.update_allnode({'feat': feat})
            return feat 
        return nb_compute(g=graph, n_feats={'feat':feat, 'feat_0':feat_0, 'norm':norm})

class EglAPPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(EglAPPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = EglAPPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h
