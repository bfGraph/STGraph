import torch


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


def generate_train_mask(size: int, train_test_split: int) -> list:
    cutoff = size * train_test_split
    return [1 if i < cutoff else 0 for i in range(size)]


def generate_test_mask(size: int, train_test_split: int) -> list:
    cutoff = size * train_test_split
    return [0 if i < cutoff else 1 for i in range(size)]
