stgraph.graph
#############

Building the Graphs
===================

The graphs' logic is implemented in a CUDA file, which requires compilation using 
nvcc to generate a shared object (.so) file. This file is then utilized by the 
Python-based graph abstraction. To compile these files, execute the provided shell scripts.

**Dynamic Graphs**

1. Grant execution permissions to the script by executing ``chmod u+x build_dynamic.sh``. 
This step should be performed whenever changes are made to the shell script.

2. Run the script as shown : ``./build_dynamic.sh [graph_names]``. For instance: ``./build_dynamic.sh gpma pcsr``

**Static Graphs**

1. Grant execution permissions to the script by executing ``chmod u+x build_static.sh``. This step should be performed whenever changes are made to the shell script.

2. Run the script as shown : ``./build_static.sh``.

.. currentmodule:: stgraph.graph
.. automodule:: stgraph.graph

Base Graph Class
================

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: class.rst

    STGraphBase
    StaticGraph
    DynamicGraph