import os
import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    usrhome = os.path.expanduser('~')
    datadir = usrhome + '/.pyg'
    if not os.path.exists(datadir):
        print('Creating data dir:' + datadir)
        os.makedirs(datadir)
    path = osp.join(datadir, name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset
