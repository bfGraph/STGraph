import pytest
from stgraph.dataset import METRLADataLoader


def METRLADataCheck(metrla: METRLADataLoader):
    assert metrla.gdata["total_timestamps"] == (
        100 if not metrla._cutoff_time else metrla._cutoff_time
    )

    assert metrla.gdata["num_nodes"] == 207
    assert metrla.gdata["num_edges"] == 1722

    edges = metrla.get_edges()
    edge_weights = metrla.get_edge_weights()
    # all_targets = metrla.get_all_targets()
    # all_features = metrla.get_all_features()

    assert len(edges) == 1722
    assert len(edges[0]) == 2

    assert len(edge_weights) == 1722

    # TODO: Check targets and features list


def test_METRLADataLoader():
    metrla_1 = METRLADataLoader(verbose=True)
    metrla_2 = METRLADataLoader(
        url="https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/METRLA.json"
    )
    metrla_3 = METRLADataLoader(num_timesteps_in=8, num_timesteps_out=8)
    metrla_4 = METRLADataLoader(cutoff_time=50)
    # metrla_5 = METRLADataLoader(redownload=True)

    METRLADataCheck(metrla_1)
    METRLADataCheck(metrla_2)
    METRLADataCheck(metrla_3)
    METRLADataCheck(metrla_4)

    with pytest.raises(TypeError) as exec:
        METRLADataLoader(num_timesteps_in="num_timesteps_in")
    assert str(exec.value) == "num_timesteps_in must be of type int"

    with pytest.raises(ValueError) as exec:
        METRLADataLoader(num_timesteps_in=-1)
    assert str(exec.value) == "num_timesteps_in must be a positive integer"

    with pytest.raises(TypeError) as exec:
        METRLADataLoader(num_timesteps_out="num_timesteps_out")
    assert str(exec.value) == "num_timesteps_out must be of type int"

    with pytest.raises(ValueError) as exec:
        METRLADataLoader(num_timesteps_out=-1)
    assert str(exec.value) == "num_timesteps_out must be a positive integer"

    with pytest.raises(TypeError) as exec:
        METRLADataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        METRLADataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"
