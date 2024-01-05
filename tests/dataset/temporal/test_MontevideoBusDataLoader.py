import pytest
from stgraph.dataset import MontevideoBusDataLoader


def MontevideoBusDataCheck(monte: MontevideoBusDataLoader):
    assert monte.gdata["total_timestamps"] == (
        744 if not monte._cutoff_time else monte._cutoff_time
    )

    assert monte.gdata["num_nodes"] == 675
    assert monte.gdata["num_edges"] == 690

    edges = monte.get_edges()
    edge_weights = monte.get_edge_weights()
    all_targets = monte.get_all_targets()
    all_features = monte.get_all_features()

    assert len(edges) == 690

    for edge in edges:
        assert len(edge) == 2

    assert len(edge_weights) == 690

    assert all_features.shape == (
        monte.gdata["total_timestamps"] - monte._lags,
        monte.gdata["num_nodes"],
        monte._lags,
    )

    assert all_targets.shape == (
        monte.gdata["total_timestamps"] - monte._lags,
        monte.gdata["num_nodes"],
    )


def test_MontevideoBusDataLoader():
    monte_1 = MontevideoBusDataLoader(verbose=True)
    monte_2 = MontevideoBusDataLoader(
        url="https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/montevideobus.json"
    )
    monte_3 = MontevideoBusDataLoader(lags=6)
    monte_4 = MontevideoBusDataLoader(cutoff_time=50)
    # monte_5 = MontevideoBusDataLoader(redownload=True)

    MontevideoBusDataCheck(monte_1)
    MontevideoBusDataCheck(monte_2)
    MontevideoBusDataCheck(monte_3)
    MontevideoBusDataCheck(monte_4)

    with pytest.raises(TypeError) as exec:
        MontevideoBusDataLoader(lags="lags")
    assert str(exec.value) == "lags must be of type int"

    with pytest.raises(ValueError) as exec:
        MontevideoBusDataLoader(lags=-1)
    assert str(exec.value) == "lags must be a positive integer"

    with pytest.raises(TypeError) as exec:
        MontevideoBusDataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        MontevideoBusDataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"

    with pytest.raises(ValueError) as exec:
        MontevideoBusDataLoader(cutoff_time=4)
    assert str(exec.value) == "cutoff_time must be greater than lags"
