import pytest
from stgraph.dataset import WikiMathDataLoader


def WikiMathDataCheck(wiki: WikiMathDataLoader):
    assert wiki.gdata["total_timestamps"] == (
        731 if not wiki._cutoff_time else wiki._cutoff_time
    )

    assert wiki.gdata["num_nodes"] == 1068
    assert wiki.gdata["num_edges"] == 27079

    edges = wiki.get_edges()
    edge_weights = wiki.get_edge_weights()
    all_targets = wiki.get_all_targets()

    assert len(edges) == 27079

    for edge in edges:
        assert len(edge) == 2

    assert len(edge_weights) == 27079

    # TODO: Add tests for features and targets arrays


def test_WikiMathDataLoader():
    wiki_1 = WikiMathDataLoader(verbose=True)
    wiki_2 = WikiMathDataLoader(redownload=True)
    wiki_3 = WikiMathDataLoader(lags=4)
    wiki_4 = WikiMathDataLoader(cutoff_time=500)

    WikiMathDataCheck(wiki_1)
    WikiMathDataCheck(wiki_2)
    WikiMathDataCheck(wiki_3)
    WikiMathDataCheck(wiki_4)

    with pytest.raises(TypeError) as exec:
        WikiMathDataLoader(lags="lags")
    assert str(exec.value) == "lags must be of type int"

    with pytest.raises(ValueError) as exec:
        WikiMathDataLoader(lags=-1)
    assert str(exec.value) == "lags must be a positive integer"

    with pytest.raises(TypeError) as exec:
        WikiMathDataLoader(cutoff_time="time")
    assert str(exec.value) == "cutoff_time must be of type int"

    with pytest.raises(ValueError) as exec:
        WikiMathDataLoader(cutoff_time=-1)
    assert str(exec.value) == "cutoff_time must be a positive integer"
