"""Represent graphs in STGraph using this abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class STGraphBase(ABC):
    r"""Represent graphs in STGraph using this abstract base class."""

    def __init__(self: STGraphBase) -> None:
        r"""Represent graphs in STGraph using this abstract base class.

        This abstract class outlines the interface for defining different
        types of graphs used in STGraph. It provides the basic structure
        and methods for graph classes. Subclasses should implement the
        abstract methods to provide specific graph functionality.

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
    def _get_graph_csr_ptrs(self: STGraphBase) -> None:
        r"""TODO:."""
        pass

    @abstractmethod
    def get_num_nodes(self: STGraphBase) -> int:
        r"""Return the number of nodes in the graph."""
        pass

    @abstractmethod
    def get_num_edges(self: STGraphBase) -> int:
        r"""Return the number of edges in the graph."""
        pass

    @abstractmethod
    def get_ndata(self: STGraphBase, field: str) -> any:
        r"""Return the graph metadata."""
        pass

    @abstractmethod
    def set_ndata(self: STGraphBase, field: str, val: any) -> None:
        r"""Set the graph metadata."""
        pass

    @property
    @abstractmethod
    def graph_type(self: STGraphBase) -> str:
        r"""Return the graph type."""
        pass
