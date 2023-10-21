"""Base class for all STGraph dataset loaders"""

import os
import json
import urllib.request

from abc import ABC, abstractmethod
from rich.console import Console

console = Console()


class STGraphDataset(ABC):
    r"""Abstract base class for graph dataset loaders

    Attributes
    ----------

    name : str
        The name of the dataset
    gdata : dict
        Meta data associated with the dataset

    _dataset : dict
        The loaded graph dataset
    _url : str
        The URL from where the dataset is downloaded online
    _verbose : bool
        Flag to control whether to display verbose info
    _cache_folder : str
        Folder inside ~/.stgraph where the dataset cache is stored
    _cache_file_type : str
        The file type used for storing the cached dataset

    Methods
    -------

    _has_dataset_cache()
        Checks if the dataset is stored in cache

    _get_cache_file_path()
        Returns the absolute path of the cached dataset file

    _init_graph_data()
        Initialises the ``gdata`` attribute with all necessary meta data

    _process_dataset()
        Processes the dataset to be used by STGraph

    _download_dataset()
        Downloads the dataset using the URL

    _save_dataset()
        Saves the dataset to cache

    _load_dataset()
        Loads the dataset from cache
    """

    def __init__(self) -> None:
        self.name = ""
        self.gdata = {}

        self._dataset = {}
        self._url = ""
        self._verbose = False
        self._cache_folder = "/dataset_cache/"
        self._cache_file_type = "json"

    def _has_dataset_cache(self) -> bool:
        r"""Checks if the dataset is stored in cache

        This private method checks whether the graph dataset cache file exists
        in the dataset cache folder. The cache .json file is found in the following
        directory ``~/.stgraph/dataset_cache/.

        Returns
        -------
        bool
            ``True`` if the cache file exists, else ``False``

        Notes
        -----
        The cache file is usually stored as a json file named as ``dataset_name.json`` and is stored
        inside the ``~/.stgraph/dataset_cache/``. Incase the directory does not exists, it
        is created by this method.

        This private method is intended for internal use within the class and should not be
        called directly from outside the class.

        Example
        -------

        .. code-block:: python

            if self._has_dataset_cache():
                # The dataset is cached, continue cached operations
            else:
                # The dataset is not cached, continue load and save operations
        """

        user_home_dir = os.path.expanduser("~")
        stgraph_dir = user_home_dir + "/.stgraph"
        cache_dir = stgraph_dir + self._cache_folder

        if os.path.exists(stgraph_dir) == False:
            os.system("mkdir " + stgraph_dir)

        if os.path.exists(cache_dir) == False:
            os.system("mkdir " + cache_dir)

        cache_file_name = self.name + "." + self._cache_file_type

        return os.path.exists(cache_dir + cache_file_name)

    def _get_cache_file_path(self) -> str:
        r"""Returns the absolute path of the cached dataset file

        Returns
        -------
        str
            The absolute path of the cached dataset file
        """

        user_home_dir = os.path.expanduser("~")
        stgraph_dir = user_home_dir + "/.stgraph"
        cache_dir = stgraph_dir + self._cache_folder
        cache_file_name = self.name + "." + self._cache_file_type

        return cache_dir + cache_file_name

    @abstractmethod
    def _init_graph_data(self) -> None:
        r"""Initialises the ``gdata`` attribute with all necessary meta data

        This is an abstract method that is implemented by ``STGraphStaticDataset``.
        The meta data is initialised based on the type of the graph dataset. The values
        are calculated as key-value pairs by the respective dataloaders when they
        are initialised.
        """
        pass

    @abstractmethod
    def _process_dataset(self) -> None:
        r"""Processes the dataset to be used by STGraph

        This is an abstract method that is to be implemented by each dataset loader.
        The implementation in specific to the nature of the dataset itself. The dataset is
        processed in such a way that it can be smoothly used within STGraph.
        """
        pass

    def _download_dataset(self) -> None:
        r"""Downloads the dataset using the URL

        Downloads the dataset files from the URL set by default for each data loader or
        by one provided by the user. If verbose mode is enabled, it displays download status.
        """
        if self._verbose:
            console.log(
                f"[cyan bold]{self.name}[/cyan bold] not present in cache. Downloading right now."
            )

        self._dataset = json.loads(urllib.request.urlopen(self._url).read())

        if self._verbose:
            console.log(f"[cyan bold]{self.name}[/cyan bold] download complete.")

    def _save_dataset(self) -> None:
        r"""Saves the dataset to cache

        Saves the downloaded dataset file to the cache folder. If verbose mode is enabled,
        it displays the save information.
        """
        with open(self._get_cache_file_path(), "w") as cache_file:
            json.dump(self._dataset, cache_file)

            if self._verbose:
                console.log(
                    f"[cyan bold]{self.name}[/cyan bold] dataset saved to cache"
                )

    def _load_dataset(self) -> None:
        r"""Loads the dataset from cache

        Loads the caches dataset json file as a python dictionary. If verbose mode is enabled,
        it displays the loading status.
        """
        if self._verbose:
            console.log(f"Loading [cyan bold]{self.name}[/cyan bold] from cache")

        with open(self._get_cache_file_path()) as cache_file:
            self._dataset = json.load(cache_file)

        if self._verbose:
            console.log(
                f"Successfully loaded [cyan bold]{self.name}[/cyan bold] from cache"
            )
