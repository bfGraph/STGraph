import pytest
from stgraph.dataset import WindmillOutputDataLoader


def WindmillOutputDataCheck(wind: WindmillOutputDataLoader):
    assert wind.gdata["total_timestamps"] == (
        17472 if not wind._cutoff_time else wind._cutoff_time
    )

    if wind._size == "large":
        assert wind.gdata["num_nodes"] == 319
        assert wind.gdata["num_edges"] == 101761
    elif wind._size == "medium":
        assert wind.gdata["num_nodes"] == 26
        assert wind.gdata["num_edges"] == 676
    elif wind._size == "small":
        assert wind.gdata["num_nodes"] == 11
        assert wind.gdata["num_edges"] == 121

    edges = wind.get_edges()
    edge_weights = wind.get_edge_weights()
    all_targets = wind.get_all_targets()

    if wind._size == "large":
        assert len(edges) == 101761
        assert len(edge_weights) == 101761
    elif wind._size == "medium":
        assert len(edges) == 676
        assert len(edge_weights) == 676
    elif wind._size == "small":
        assert len(edges) == 121
        assert len(edge_weights) == 121

    for edge in edges:
        len(edge) == 2

    # TODO: Test for targets and features


def test_WindmillOutputDataLoader():
    for size in ["large", "medium", "small"]:
        wind_1 = WindmillOutputDataLoader(verbose=True, size=size)
        wind_2 = WindmillOutputDataLoader(redownload=True, size=size)
        wind_3 = WindmillOutputDataLoader(lags=4, size=size)
        wind_4 = WindmillOutputDataLoader(cutoff_time=100, size=size)

        WindmillOutputDataCheck(wind_1)
        WindmillOutputDataCheck(wind_2)
        WindmillOutputDataCheck(wind_3)
        WindmillOutputDataCheck(wind_4)

        with pytest.raises(TypeError) as exec:
            WindmillOutputDataLoader(lags="lags", size=size)
        assert str(exec.value) == "lags must be of type int"

        with pytest.raises(ValueError) as exec:
            WindmillOutputDataLoader(lags=-1, size=size)
        assert str(exec.value) == "lags must be a positive integer"

        with pytest.raises(TypeError) as exec:
            WindmillOutputDataLoader(cutoff_time="time", size=size)
        assert str(exec.value) == "cutoff_time must be of type int"

        with pytest.raises(ValueError) as exec:
            WindmillOutputDataLoader(cutoff_time=-1, size=size)
        assert str(exec.value) == "cutoff_time must be a positive integer"

    with pytest.raises(TypeError) as exec:
        WindmillOutputDataLoader(size=1)
    assert str(exec.value) == "size must be of type string"

    with pytest.raises(ValueError) as exec:
        WindmillOutputDataLoader(size="big")
    assert (
        str(exec.value) == "size must take either of the following values : "
        "large, medium or small"
    )
