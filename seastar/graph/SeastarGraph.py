from abc import ABC, abstractmethod

class SeastarGraph(ABC):
    def __init__(self):
        self._ndata = {}
        
        self._forward_graph = None
        self._backward_graph = None
        
        self.fwd_row_offset_ptr = None
        self.fwd_column_indices_ptr = None
        self.fwd_eids_ptr = None
        
        self.bwd_row_offset_ptr = None
        self.bwd_column_indices_ptr = None
        self.bwd_eids_ptr = None
    
    @abstractmethod
    def _get_graph_csr_ptrs(self):
        pass
    
    @abstractmethod
    def get_num_nodes(self):
        pass
    
    @abstractmethod
    def get_num_edges(self):
        pass

    @abstractmethod
    def get_ndata(self, field):
        pass

    @abstractmethod
    def set_ndata(self, field, val):
        pass
    
    @property
    @abstractmethod
    def graph_type(self):
        pass