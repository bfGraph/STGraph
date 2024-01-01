import numpy as np
import urllib.request

from stgraph.dataset import EnglandCovidDataLoader


def EnglandCovidDataCheck(eng_covid: EnglandCovidDataLoader):
    # test for gdata
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

    # test for features and targets
    # TODO:


def test_EnglandCovidDataLoader():
    eng_covid = EnglandCovidDataLoader()
    eng_covid_1 = EnglandCovidDataLoader(cutoff_time=30)
    eng_covid_2 = EnglandCovidDataLoader(
        url="https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
    )

    EnglandCovidDataCheck(eng_covid)
    EnglandCovidDataCheck(eng_covid_1)
