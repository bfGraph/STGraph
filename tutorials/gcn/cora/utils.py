import torch
from stgraph.graph import StaticGraph


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


# GPU | CPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def to_default_device(data):
    if isinstance(data, (list, tuple)):
        return [to_default_device(x, get_default_device()) for x in data]

    return data.to(get_default_device(), non_blocking=True)


def generate_train_mask(size: int, train_test_split: float) -> list:
    cutoff = size * train_test_split
    return [1 if i < cutoff else 0 for i in range(size)]


def generate_test_mask(size: int, train_test_split: float) -> list:
    cutoff = size * train_test_split
    return [0 if i < cutoff else 1 for i in range(size)]


def row_normalize_feature(mx):
    """Row-normalize PyTorch tensor"""
    # Compute the sum of each row
    rowsum = mx.sum(dim=1, keepdim=True)

    # Compute the inverse of the row sums, handling division by zero
    r_inv = torch.where(rowsum != 0, 1.0 / rowsum, torch.zeros_like(rowsum))

    # Perform the row normalization
    mx = mx * r_inv

    return mx


def get_node_norms(graph: StaticGraph):
    degrees = torch.from_numpy(graph.weighted_in_degrees()).type(torch.int32)
    norm = torch.pow(degrees, -0.5)
    norm[torch.isinf(norm)] = 0
    return to_default_device(norm).unsqueeze(1)
