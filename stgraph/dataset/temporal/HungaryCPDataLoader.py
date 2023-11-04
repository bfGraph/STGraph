from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class HungaryCPDataLoader(STGraphTemporalDataset):
    def __init__(self, verbose=False, url=None, lags=4, cutoff_time=None) -> None:
        r"""County level chicken pox cases in Hungary"""

        super.__init__()

        self.name = "Hungary Chickenpox"
        self._verbose = verbose
        self._lags = lags

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/HungaryCP.json"
        else:
            self._url = url

        if self._has_dataset_cache():
            self._load_dataset()
        else:
            self._download_dataset()
            self._save_dataset()

        self._process_dataset()
