from abc import ABC, abstractmethod

class SeastarGraph(ABC):
    def __init__(self):
        self.ndata = {}
        self.num_nodes = 0
        self.num_edges = 0
        
        self.row_offset_ptr = None
        self.column_indices_ptr = None
        self.eids_ptr = None
    
    @property
    @abstractmethod
    def graph_type(self):
        pass