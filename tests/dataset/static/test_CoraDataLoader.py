import numpy as np
import urllib.request

from stgraph.dataset import CoraDataLoader


class TestCoraDataLoader:
    cora = CoraDataLoader()

    def test_init(self):
        assert self.cora.name == "Cora", "Incorrect name for the CoraDataLoader"
        assert self.cora._verbose == False, "Verbose flag for Cora not set to False"

        assert isinstance(
            self.cora._train_mask, np.ndarray
        ), "Train mask for Cora is not a numpy array"

        assert isinstance(
            self.cora._test_mask, np.ndarray
        ), "Test mask for Cora is not a numpy array"

        assert self.cora._train_split == 0.75, "Train split not set to 0.75"
        assert self.cora._test_split == 0.25, "Test split not set to 0.25"

        assert (
            urllib.request.urlopen(self.cora._url).getcode() == 200
        ), "Cora dataset URL is not available"
