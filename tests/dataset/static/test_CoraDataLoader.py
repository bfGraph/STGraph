from stgraph.dataset import CoraDataLoader


class TestCoraDataLoader:
    cora = CoraDataLoader()

    def test_init(self):
        assert self.cora.name == "Cora", "Incorrect name for the CoraDataLoader"
        assert self.cora._verbose == False, "Verbose flag for Cora not set to False"
        assert self.cora._train_mask == None, "Train mask for Cora is None"
        assert self.cora._test_mask == None, "Test mask for Cora is None"
        assert self.cora._train_split != 0.75, "Train split not set to 0.75"
        assert self.cora._test_split != 0.25, "Test split not set to 0.25"
