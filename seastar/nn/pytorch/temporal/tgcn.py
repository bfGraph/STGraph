import torch
from seastar.nn.pytorch.graph_conv import GraphConv


class TGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Update GCN Layer
        self.conv_z = GraphConv(self.in_channels, self.out_channels, activation=None)
       
        # Update linear layer
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Reset GCN layer
        self.conv_r = GraphConv(self.in_channels, self.out_channels, activation=None)

        # Reset linear layer
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

        # Candidate (Current Memory Content) GCN layer
        self.conv_h = GraphConv(self.in_channels, self.out_channels, activation=None)

        # Candidate linear layer
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, g, X, edge_weight, H):
        h = self.conv_z(g, X, edge_weight=edge_weight)
        Z = torch.cat((h, H), dim=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, g, X, edge_weight, H):
        h = self.conv_r(g, X, edge_weight=edge_weight)
        R = torch.cat((h, H), dim=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, g, X, edge_weight, H, R):
        h = self.conv_h(g, X, edge_weight=edge_weight)
        H_tilde = torch.cat((h, H * R), dim=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, g, X, edge_weight=None, H=None):

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(g, X, edge_weight, H)
        R = self._calculate_reset_gate(g, X, edge_weight, H)
        H_tilde = self._calculate_candidate_state(g, X, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
