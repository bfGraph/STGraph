import pytest
from stgraph.dataset import PedalMeDataLoader


def PedalMeDataCheck(pedal: PedalMeDataLoader):
    assert pedal.gdata["total_timestamps"] == (
        36 if not pedal._cutoff_time else pedal._cutoff_time
    )

    assert pedal.gdata["num_nodes"] == 15
    assert pedal.gdata["num_edges"] == 225

    edges = pedal.get_edges()
    edge_weights = pedal.get_edge_weights()
    all_targets = pedal.get_all_targets()

    assert len(edges) == 225

    for edge in edges:
        assert len(edge) == 2

    assert len(edge_weights) == 225

    assert all_targets.shape == (
        pedal.gdata["total_timestamps"] - pedal._lags,
        pedal.gdata["num_nodes"],
    )


def test_PedalMeDataLoader():
    pedal_1 = PedalMeDataLoader(verbose=True)
    pedal_2 = PedalMeDataLoader(redownload=True)
    pedal_3 = PedalMeDataLoader(lags=6)
    pedal_4 = PedalMeDataLoader(cutoff_time=20)

    PedalMeDataCheck(pedal_1)
    PedalMeDataCheck(pedal_2)
    PedalMeDataCheck(pedal_3)
    PedalMeDataCheck(pedal_4)

    with pytest.raises(TypeError) as exec:
        PedalMeDataLoader(lags="lags")
    assert str(exec.value) == "lags must be of type int"

    with pytest.raises(ValueError) as exec:
        PedalMeDataLoader(lags=-1)
    assert str(exec.value) == "lags must be a positive integer"

    with pytest.raises(TypeError) as exec:
        PedalMeDataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        PedalMeDataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"

    with pytest.raises(ValueError) as exec:
        PedalMeDataLoader(cutoff_time=4)
    assert str(exec.value) == "cutoff_time must be greater than lags"
