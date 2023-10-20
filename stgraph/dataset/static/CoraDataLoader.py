import json
import urllib.request

from stgraph.dataset.static.STGraphStaticDataset import STGraphStaticDataset

from rich.console import Console

console = Console()


class CoraDataLoader(STGraphStaticDataset):
    def __init__(self, verbose=False, url=None) -> None:
        super().__init__()

        self.name = "Cora"
        self._verbose = verbose

        # setting the dataset URL. It is either provided by the user or the
        # default STGraph-Datasets repo link is taken
        if not url:
            self._url = "https://raw.githubusercontent.com/bfGraph/STGraph-Datasets/main/cora.json"
        else:
            self._url = url

        if self._has_dataset_cache():
            self._load_dataset()
        else:
            self._download_dataset()
            self._save_dataset()
