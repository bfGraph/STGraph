import os
import json
from rich.console import Console
import numpy as np
console = Console()

from rich import inspect

class MontevideoBusDataLoader:
    def __init__(self, cutoff_time, lags: int = 4, verbose: bool = False, for_seastar: bool = False):
        self.name = "MontevideoBus"
        self._local_path =  f'../../dataset/MontevideoBus/{self.name}.json'
        self._verbose = verbose
        self.for_seastar = for_seastar
        self.lags = lags
    
        self._load_dataset()
    
    def _load_dataset(self):
        if os.path.exists(self._local_path):
            dataset_file = open(self._local_path)
            self._dataset = json.load(dataset_file)
            
            inspect(self._dataset)
            quit()
            
            if self._verbose:
                console.log(f'Loading [cyan]{self.name}[/cyan] dataset from dataset/{self.name}.json')
        else:
            console.log(f'Failed to find [cyan]{self.name}[/cyan] dataset from dataset')
            quit()
            
mvb = MontevideoBusDataLoader(0)