"""Base class for all STGraph dataset loaders"""

import os

from abc import ABC, abstractmethod


class STGraphDataset(ABC):
    def __init__(self) -> None:
        self.name = "Cora"
        self.gdata = {}

        self._url = None
        self._verbose = None
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

        Example
        -------

        .. code-block:: python

            if self._has_dataset_cache():
                # The dataset is cached, continue cached operations
            else:
                # The dataset is not cached, continue load and save operations

        Warning
        -------
        This private method is intended for internal use within the class and should not be
        called directly from outside the class.
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
