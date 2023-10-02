from rich.console import Console
from rich.table import Table

console = Console()

class BenchmarkTable:
    def __init__(self, title: str, col_name_list: list[str]):
        self.title = '\n' + title + '\n'
        self.col_name_list = col_name_list
        self._table = Table(title=self.title, show_edge=False, style="black bold")
        self._num_cols = len(col_name_list)
        self._num_rows = 0
        
        self._table_add_columns()
        
    def _table_add_columns(self):
        for col_name in self.col_name_list:
            self._table.add_column(col_name, justify="left")
            
    def add_row(self, values: list):
        values_str = tuple([str(val) for val in values])
        self._table.add_row(*values_str)
    
    def display(self):
        console.print(self._table)