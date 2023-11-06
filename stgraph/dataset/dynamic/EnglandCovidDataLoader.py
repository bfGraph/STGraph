from stgraph.dataset.dynamic.STGraphDynamicDataset import STGraphDynamicDataset


class EnglandCovidDataLoader(STGraphDynamicDataset):
    def __init__(
        self,
        verbose: bool = False,
        url: str = None,
        lags: int = 8,
        cutoff_time: int = None,
    ) -> None:
        super().__init__()

        self.name = "England COVID"
        self._verbose = verbose
        self._lags = lags
        self._cutoff_time = cutoff_time

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

    def _process_dataset(self) -> None:
        pass

    def _get_total_timestamps(self) -> None:
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                len(self._dataset["time_periods"]), self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = len(self._dataset["time_periods"])
