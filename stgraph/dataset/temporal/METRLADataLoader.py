import numpy as np

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset


class METRLADataLoader(STGraphTemporalDataset):
    def __init__(
        self,
        verbose=True,
        url=None,
        num_timesteps_in=12,
        num_timesteps_out=12,
        cutoff_time=None,
    ):
        r"""A traffic forecasting dataset based on Los Angeles Metropolitan traffic conditions."""

        super().__init__()

        assert (
            num_timesteps_in > 0 and type(num_timesteps_in) == int
        ), "Invalid num_timesteps_in value"
        assert (
            num_timesteps_out > 0 and type(num_timesteps_out) == int
        ), "Invalid num_timesteps_out value"

        self.name = "METRLA"
        self._verbose = verbose
        self._num_timesteps_in = num_timesteps_in
        self._num_timesteps_out = num_timesteps_out
        self._cutoff_time = cutoff_time

        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/METRLA.json"
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

    def _set_total_timestamps(self) -> None:
        r"""Sets the total timestamps present in the dataset

        It sets the total timestamps present in the dataset into the
        gdata attribute dictionary. It is the minimum of the cutoff time
        choosen by the user and the total time periods present in the
        original dataset.
        """
        if self._cutoff_time != None:
            self.gdata["total_timestamps"] = min(
                self._dataset["time_periods"], self._cutoff_time
            )
        else:
            self.gdata["total_timestamps"] = self._dataset["time_periods"]
