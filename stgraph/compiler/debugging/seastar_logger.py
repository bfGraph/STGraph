from prettytable import PrettyTable
# from prettytable import SINGLE_BORDER, ALL

# TODO: This is manually taken from the prettytable page since there is an import error
SINGLE_BORDER = 16
ALL = 1

from rich.console import Console
console = Console()

import stgraph.compiler.debugging.print_variables as print_var

def print_log(log_message):
    if print_var.is_print_verbose_log:
        console.log(log_message)