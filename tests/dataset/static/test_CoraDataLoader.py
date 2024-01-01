import numpy as np
import urllib.request

from stgraph.dataset import CoraDataLoader


def CoraDataCheck(cora: CoraDataLoader):
    assert len(cora._edge_list) == 10556
    assert cora._all_features.shape == (2708, 1433)
    assert cora._all_targets.shape == (2708,)

    assert cora.gdata["num_nodes"] == 2708
    assert cora.gdata["num_edges"] == 10556
    assert cora.gdata["num_feats"] == 1433
    assert cora.gdata["num_classes"] == 7

    edge_list = cora.get_edges()

    assert len(edge_list) == 10556 and len(edge_list[0]) == 2
    assert cora.get_all_features().shape == (2708, 1433)
    assert cora.get_all_targets().shape == (2708,)


def test_CoraDataLoader():
    cora = CoraDataLoader()

    cora_1 = CoraDataLoader(
        url="https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/cora.json",
    )

    CoraDataCheck(cora)
    CoraDataCheck(cora_1)
