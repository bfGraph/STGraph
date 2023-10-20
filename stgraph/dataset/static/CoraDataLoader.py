from stgraph.dataset.static.STGraphStaticDataset import STGraphStaticDataset


class CoraDataLoader(STGraphStaticDataset):
    def __init__(self) -> None:
        super().__init__()

        self.name = "Cora"

        if self._has_dataset_cache():
            print("🪄 Cached Dataset Exists")
        else:
            print("⚠️ Cached Dataset doesn't exist")
