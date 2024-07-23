r"""Constants to be used across the project."""

from enum import Enum


class SizeConstants(Enum):
    r"""Data Size Related Constants.

    +----------------+-------+----------------------------------------------+
    | Constant       | Value | Description                                  |
    +================+=======+==============================================+
    | NODE_NORM_SIZE | 2     | Length of the node-wise normalization tensor |
    +----------------+-------+----------------------------------------------+

    """
    NODE_NORM_SIZE = 2
