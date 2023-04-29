from abc import ABC, abstractmethod

class SeastarGraph(ABC):
    def __init__(self):
        self.ndata = {}
        self.num_nodes = 0
        self.num_edges = 0
        
        self.fwd_row_offset_ptr = None
        self.fwd_column_indices_ptr = None
        self.fwd_eids_ptr = None
        
        self.bwd_row_offset_ptr = None
        self.bwd_column_indices_ptr = None
        self.bwd_eids_ptr = None
    
    @abstractmethod
    def _get_graph_csr_ptrs(self):
        pass
    
    @property
    @abstractmethod
    def graph_type(self):
        pass