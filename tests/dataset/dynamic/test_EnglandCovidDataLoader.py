import pytest
from stgraph.dataset.dynamic.EnglandCovidDataLoader import EnglandCovidDataLoader


class TestEnglandCovidDataLoader:
    @pytest.fixture
    def eng_data(self):
        return EnglandCovidDataLoader()

    def test_constructor(self, eng_data: EnglandCovidDataLoader):
        assert eng_data.name == "England COVID", "There has been a name change"
        assert eng_data._verbose == False, "Default value for _verbose must be False"
        assert eng_data._lags == 8, "Default value for _lags must be 8"
        assert (
            eng_data._cutoff_time == None
        ), "Default value for _cutoff_time must be None"
        assert (
            eng_data._url
            == "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/england_covid.json"
        ), "Default URL for the dataset has changed"

    def test_graph_data(self, eng_data: EnglandCovidDataLoader):
        assert eng_data.gdata["total_timestamps"] == 61
        assert eng_data.gdata["num_edges"] == {
            "0": 2158,
            "1": 1743,
            "2": 1521,
            "3": 2108,
            "4": 2063,
            "5": 2017,
            "6": 1986,
            "7": 1949,
            "8": 1377,
            "9": 1160,
            "10": 1826,
            "11": 1616,
            "12": 1503,
            "13": 1466,
            "14": 1404,
            "15": 975,
            "16": 843,
            "17": 1341,
            "18": 1346,
            "19": 1347,
            "20": 1344,
            "21": 1316,
            "22": 964,
            "23": 871,
            "24": 1355,
            "25": 1331,
            "26": 1343,
            "27": 1349,
            "28": 1049,
            "29": 945,
            "30": 836,
            "31": 955,
            "32": 1346,
            "33": 1348,
            "34": 1371,
            "35": 1343,
            "36": 973,
            "37": 885,
            "38": 1356,
            "39": 1366,
            "40": 1399,
            "41": 1391,
            "42": 1375,
            "43": 1016,
            "44": 928,
            "45": 1417,
            "46": 1415,
            "47": 1425,
            "48": 1420,
            "49": 1398,
            "50": 1048,
            "51": 950,
            "52": 1424,
            "53": 1483,
            "54": 1481,
            "55": 1466,
            "56": 1125,
            "57": 1062,
            "58": 924,
            "59": 1476,
            "60": 1511,
        }

    def test_get_edges(self, eng_data: EnglandCovidDataLoader):
        edges = eng_data.get_edges()

        assert (
            len(edges) == eng_data.gdata["total_timestamps"]
        ), "The number of elements in the edge list doesn't match with the total_timestamps"
