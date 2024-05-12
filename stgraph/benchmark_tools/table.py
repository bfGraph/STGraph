"""Table that can display benchmarking data and other info."""

from __future__ import annotations

from typing import IO

from rich.console import Console
from rich.table import Table


class BenchmarkTable:
    r"""Table that can display benchmarking data and other info.

    This class provides functionality to create and display tables for
    benchmarking data along with other relevant information.

    Example:
    --------
    .. code-block:: python

        from stgraph.benchmark_tools import BenchmarkTable

        table = BenchmarkTable(
            title = "GCN Benchmark Data",
            col_name_list = ["Model", "Time", "MSE"]
        )

        table.add_row("GCN", 12.56, 45.89)
        table.add_row("GCN", 23.34, 44.32)

        table.display()

    Parameters
    ----------
    title : str
        The title of the table
    col_name_list : list[str]
        A list of the table column names

    Attributes
    ----------
    title : str
        The title of the table
    col_name_list : list[str]
        A list of the table column names

    """

    def __init__(self: BenchmarkTable, title: str, col_name_list: list[str]) -> None:
        r"""Table that can display benchmarking data and other info."""
        self.title = "\n" + title + "\n"
        self.col_name_list = col_name_list
        self._table = Table(title=self.title, show_edge=False, style="black bold")
        self._num_cols = len(col_name_list)
        self._num_rows = 0

        self._table_add_columns()

    def _table_add_columns(self: BenchmarkTable) -> None:
        r"""Prepare the table by adding all the columns."""
        for col_name in self.col_name_list:
            self._table.add_column(col_name, justify="left")

    def add_row(self: BenchmarkTable, values: list) -> None:
        r"""Add a row of data to the table.

        Parameters
        ----------
        values : list
            A list of values for each column in the row.

        """
        values_str = tuple([str(val) for val in values])
        self._table.add_row(*values_str)

    def display(self: BenchmarkTable, output_file: IO[str] | None = None) -> None:
        r"""Display entire table with data.

        Parameters
        ----------
        output_file : Optional[IO[str]], optional
            File object to write the table to.

        """
        console = Console() if not output_file else Console(file=output_file)
        console.print(self._table)
