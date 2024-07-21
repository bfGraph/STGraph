"""Utility methods for GCN training on Cora dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from stgraph.graph import StaticGraph


def accuracy(logits: Tensor, labels: Tensor) -> float:
    r"""Compute the accuracy of the predictions.

    Parameters
    ----------
    logits : Tensor
        The predicted output from the model, of shape (num_samples, num_classes).
    labels : Tensor
        The ground truth labels, of shape (num_samples,).

    Returns
    -------
    float :
        The accuracy of the predictions, calculated as the proportion of
        correct predictions out of the total number of samples.

    """
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


# GPU | CPU
def get_default_device() -> torch.device:
    r"""Return the default device to be used for tensor operations.

    Checks if CUDA is available and returns the first GPU device if it is;
    otherwise, returns the CPU device.

    Returns
    -------
    torch.device :
        The default device ("cuda:0" if available, otherwise "cpu").

    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def to_default_device(data: Tensor | list | tuple) -> Tensor | list | tuple:
    r"""Move the given data to the default device.

    If the data is a list or tuple, recursively moves each element to the default device.
    Otherwise, moves the data directly to the default device.

    Parameters
    ----------
    data : Tensor | list | tuple
        The data to be moved to the default device

    Returns
    -------
    Tensor | list | tuple :
        The data that is moved to the default device

    """
    if isinstance(data, (list, tuple)):
        return [to_default_device(x) for x in data]

    return data.to(get_default_device(), non_blocking=True)


def generate_train_mask(size: int, train_test_split: float) -> list:
    r"""Generate a mask for training data.

    Creates a binary mask where the first portion, determined by ``train_test_split``, is set to 1
    (indicating training samples) and the rest is set to 0 (indicating non-training samples).

    Parameters
    ----------
    size : int
        The total number of samples.
    train_test_split : float
        Fraction of samples used for training.

    Returns
    -------
    list :
        A binary mask where 1 represents the training sample.

    """
    cutoff = size * train_test_split
    return [1 if i < cutoff else 0 for i in range(size)]


def generate_test_mask(size: int, train_test_split: float) -> list:
    r"""Generate a mask for testing data.

    Creates a binary mask where the first portion, determined by ``train_test_split``, is set to 0
    (indicating non-testing samples) and the rest is set to 1 (indicating testing samples).

    Parameters
    ----------
    size : int
        The total number of samples.
    train_test_split : float
        Fraction of samples used for training.

    Returns
    -------
    list :
        A binary mask where 1 represents the testing sample.

    """
    cutoff = size * train_test_split
    return [0 if i < cutoff else 1 for i in range(size)]


def row_normalize_feature(features: Tensor) -> Tensor:
    """Row-normalizes the node features.

    Scales each node features such that the sum of the elements for each node feature is 1.0.
    If the sum of a row is zero, the row is normalized to zero.

    Parameters
    ----------
    features : Tensor
        The node feature tensor of shape (num_nodes, feat_size).

    Returns
    -------
    Tensor :
        The row-normalized node features.

    """
    # Compute the sum of each row
    row_sum = features.sum(dim=1, keepdim=True)

    # Compute the inverse of the row sums, handling division by zero
    r_inv = torch.where(row_sum != 0, 1.0 / row_sum, torch.zeros_like(row_sum))

    return features * r_inv


def get_node_norms(graph: StaticGraph) -> Tensor:
    r"""Compute node normalization factors for a graph.

    The normalization factor for each node is calculated as the inverse square root of its degree.
    Nodes with an infinite normalization factor (due to zero degree) are set to zero.

    Parameters
    ----------
    graph : StaticGraph
        The static graph object.

    Returns
    -------
    Tensor :
        A tensor of shape (num_nodes, 1) containing the normalization factors for each node.

    """
    degrees = torch.from_numpy(graph.weighted_in_degrees()).type(torch.int32)
    norm = torch.pow(degrees, -0.5)
    norm[torch.isinf(norm)] = 0
    return to_default_device(norm).unsqueeze(1)
