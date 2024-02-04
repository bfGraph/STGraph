import pytest

from stgraph.dataset import EnglandCovidDataLoader


def EnglandCovidDataCheck(eng_covid: EnglandCovidDataLoader):
    assert eng_covid.gdata["total_timestamps"] == (
        61 if not eng_covid._cutoff_time else eng_covid._cutoff_time
    )
    assert (
        len(list(eng_covid.gdata["num_nodes"].values()))
        == eng_covid.gdata["total_timestamps"]
    )

    for time, num_node in eng_covid.gdata["num_nodes"].items():
        assert num_node == 129

    assert (
        len(list(eng_covid.gdata["num_edges"].values()))
        == eng_covid.gdata["total_timestamps"]
    )

    edge_list = eng_covid.get_edges()

    assert len(edge_list) == eng_covid.gdata["total_timestamps"]
    assert len(edge_list[0][0]) == 2

    edge_weights = eng_covid.get_edge_weights()

    assert len(edge_weights) == eng_covid.gdata["total_timestamps"]

    for i in range(len(edge_list)):
        assert len(edge_list[i]) == len(edge_weights[i])

    all_features = eng_covid.get_all_features()
    all_targets = eng_covid.get_all_targets()

    assert len(all_features) == eng_covid.gdata["total_timestamps"] - eng_covid._lags

    assert all_features[0].shape == (
        eng_covid.gdata["num_nodes"]["0"],
        eng_covid._lags,
    )

    assert len(all_targets) == eng_covid.gdata["total_timestamps"] - eng_covid._lags

    assert all_targets[0].shape == (eng_covid.gdata["num_nodes"]["0"],)


def test_EnglandCovidDataLoader():
    eng_covid = EnglandCovidDataLoader(verbose=True)
    eng_covid_1 = EnglandCovidDataLoader(cutoff_time=30)
    eng_covid_3 = EnglandCovidDataLoader(lags=12)
    eng_covid_4 = EnglandCovidDataLoader(redownload=True)

    EnglandCovidDataCheck(eng_covid)
    EnglandCovidDataCheck(eng_covid_1)
    EnglandCovidDataCheck(eng_covid_3)
    EnglandCovidDataCheck(eng_covid_4)

    with pytest.raises(TypeError) as exec:
        EnglandCovidDataLoader(lags="lags")
    assert str(exec.value) == "lags must be of type int"

    with pytest.raises(ValueError) as exec:
        EnglandCovidDataLoader(lags=-1)
    assert str(exec.value) == "lags must be a positive integer"

    with pytest.raises(TypeError) as exec:
        EnglandCovidDataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        EnglandCovidDataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"
