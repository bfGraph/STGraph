from abc import ABC, abstractmethod


class STGraphBase(ABC):
    r"""An abstract base class used to represent graphs in STGraph

    This abstract class outlines the interface for defining different types of graphs
    used in STGraph. It provides the basic structure and methods for graph classes.
    Subclasses should implement the abstract methods to provide specific graph functionality.

    Attributes
    ----------

    fwd_row_offset_ptr
        Pointer to the forward graphs row offset array

    fwd_column_indices_ptr
        Pointer to the forward graphs column indices array

    fwd_eids_ptr
        Pointer to the forward graphs edge ID array

    fwd_node_ids_ptr
        Pointer to the forward graphs node ID array

    bwd_row_offset_ptr
        Pointer to the backward graphs row offset array

    bwd_column_indices_ptr
        Pointer to the backward graphs column indices array

    bwd_eids_ptr
        Pointer to the backward graphs edge ID array

    bwd_node_ids_ptr
        Pointer to the backward graphs node ID array

    """

    def __init__(self):
        self._ndata = {}

        self._forward_graph = None
        self._backward_graph = None

        self.fwd_row_offset_ptr = None
        self.fwd_column_indices_ptr = None
        self.fwd_eids_ptr = None
        self.fwd_node_ids_ptr = None

        self.bwd_row_offset_ptr = None
        self.bwd_column_indices_ptr = None
        self.bwd_eids_ptr = None
        self.bwd_node_ids_ptr = None

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
