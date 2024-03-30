import torch


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_default_device(data):
    if isinstance(data, (list, tuple)):
        return [to_default_device(x) for x in data]
    return data.to(get_default_device(), non_blocking=True)
