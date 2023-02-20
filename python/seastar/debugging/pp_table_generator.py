"""Generate PrettyTable tables

Used for debugging purposes by creating PrettyTable objects based
on style preferences
"""

from prettytable import PrettyTable, HEADER, NONE, SINGLE_BORDER

def genBorderlessTable() -> PrettyTable :
    """Generates a borderless PrettyTable table instance"""
    
    borderlessTable = PrettyTable()

    borderlessTable.set_style(SINGLE_BORDER)
    borderlessTable.border = True
    borderlessTable.hrules = HEADER
    borderlessTable.vrules = NONE

    return borderlessTable