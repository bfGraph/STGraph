r"""Script to train GCN on Cora dataset."""

import sys
import traceback

import torch
from model import GCN
from torch.nn.functional import cross_entropy
from utils import (
    accuracy,
    generate_test_mask,
    generate_train_mask,
    get_node_norms,
    row_normalize_feature,
)

from stgraph.utils import DataTable
from stgraph.dataset import CoraDataLoader
from stgraph.graph.static.static_graph import StaticGraph


def train(
    lr: float,
    num_epochs: int,
    num_hidden: int,
    num_hidden_layers: int,
    weight_decay: float,
) -> None:
    r"""Script to train GCN on Cora dataset.

    Parameters
    ----------
    lr : float
        Learning Rate.
    num_epochs : int
        Number of Epochs.
    num_hidden : int
        Number of hidden units in hidden layer.
    num_hidden_layers : int
        Number of hidden layers.
    weight_decay : float
        Weight decay value for L2 regularization.

    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(1)

    cora = CoraDataLoader()

    node_features = row_normalize_feature(
        torch.FloatTensor(cora.get_all_features()),
    )
    node_labels = torch.LongTensor(cora.get_all_targets())
    edge_weights = [1 for _ in range(cora.gdata["num_edges"])]

    train_mask = torch.BoolTensor(
        generate_train_mask(cora.gdata["num_nodes"], 0.7),
    )
    test_mask = torch.BoolTensor(
        generate_test_mask(cora.gdata["num_nodes"], 0.7),
    )

    torch.cuda.set_device(0)
    node_features = node_features.cuda()
    node_labels = node_labels.cuda()
    train_mask = train_mask.cuda()
    test_mask = test_mask.cuda()

    cora_graph = StaticGraph(
        edge_list=cora.get_edges(),
        edge_weights=edge_weights,
        num_nodes=cora.gdata["num_nodes"],
    )

    cora_graph.set_ndata("norm", get_node_norms(cora_graph))

    model = GCN(
        graph=cora_graph,
        in_feats=cora.gdata["num_feats"],
        n_hidden=num_hidden,
        n_classes=cora.gdata["num_classes"],
        n_hidden_layers=num_hidden_layers,
    ).cuda()

    loss_function = cross_entropy
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    table = DataTable(
        "STGraph GCN on CORA dataset",
        ["Epoch", "Train Accuracy %", "Loss"],
    )

    try:
        print("Started Training")
        for epoch in range(num_epochs):
            model.train()
            torch.cuda.synchronize()

            logits = model.forward(node_features)
            loss = loss_function(logits[train_mask], node_labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            train_acc = accuracy(logits[train_mask], node_labels[train_mask])

            table.add_row(
                [epoch, float(f"{train_acc * 100:.2f}"), float(f"{loss.item():.5f}")],
            )
        print("Training Ended")
        table.display()

        print("Evaluating trained GCN model on the Test Set")

        model.eval()
        logits_test = model(node_features)
        loss_test = loss_function(logits_test[train_mask], node_labels[train_mask])
        test_acc = accuracy(logits_test[test_mask], node_labels[test_mask])

        print(f"Loss for Test: {loss_test}")
        print(f"Accuracy for Test: {float(test_acc) * 100} %")

    except Exception as e:
        print("------------- Error -------------")
        print(e)
        traceback.print_exc()
