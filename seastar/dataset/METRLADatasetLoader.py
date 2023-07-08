import os
import json
from rich.console import Console
import numpy as np
console = Console()

class METRLADatasetLoader:
    def __init__(self, verbose: bool = False, for_seastar: bool = False):
        self.name = "METRLA"
        self._local_path = f'../../dataset/{self.name}/METRLA.json'
        self._verbose = verbose
        self.for_seastar = for_seastar