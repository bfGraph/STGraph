import pytest

from stgraph.dataset import HungaryCPDataLoader


def HungaryCPDataChecker(hungary: HungaryCPDataLoader):
    assert hungary.gdata["total_timestamps"] == (
        521 if not hungary._cutoff_time else hungary._cutoff_time
    )

    assert hungary.gdata["num_nodes"] == 20
    assert hungary.gdata["num_edges"] == 102

    edges = hungary.get_edges()
    edge_weights = hungary.get_edge_weights()
    all_targets = hungary.get_all_targets()

    assert len(edges) == 102
    assert len(edges[0]) == 2

    assert len(edge_weights) == 102

    assert len(all_targets) == hungary.gdata["total_timestamps"] - hungary._lags
    assert all_targets[0].shape == (hungary.gdata["num_nodes"],)


def test_HungaryCPDataLoader():
    hungary_1 = HungaryCPDataLoader(verbose=True)
    hungary_2 = HungaryCPDataLoader(lags=6)
    hungary_3 = HungaryCPDataLoader(cutoff_time=100)
    hungary_4 = HungaryCPDataLoader(
        url="https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/HungaryCP.json"
    )

    HungaryCPDataChecker(hungary_1)
    HungaryCPDataChecker(hungary_2)
    HungaryCPDataChecker(hungary_3)
    # HungaryCPDataChecker(hungary_4)

    with pytest.raises(TypeError) as exec:
        HungaryCPDataLoader(lags="lags")
    assert str(exec.value) == "lags must be of type int"

    with pytest.raises(ValueError) as exec:
        HungaryCPDataLoader(lags=-1)
    assert str(exec.value) == "lags must be a positive integer"

    with pytest.raises(TypeError) as exec:
        HungaryCPDataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        HungaryCPDataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"
