===================================
Welcome to Seastar's documentation!
===================================

This documentation was made to understand how the Seastar compiler works.
Feel free to use it for your project and learning purposes.


What is Seastar?
================

Seastar is a system for programming GNN models using a vertex-centric approach. This new programming approach 
ensures that the logic for various GNN models can be written intuitively. It also allows those reading 
the code to ascertain the models purpose. The Seastar system can be used to implement models like GCN and GAT
with relative ease.

While Seastar provides a new programming approach, it can perform optimizations on the model to bridge some of the gaps in
other GNN frameworks like DGL and PyG. Seastar takes a function representing the operations performed on a single 
vertex as input.

Seastar is successful in achieving lesser memory consumption and faster execution in comparison to PyG and DGL.

.. toctree::
   :maxdepth: 1
   :caption: Codebase

   exp
   python

.. toctree::
   :maxdepth: 1
   :caption: Learning Seastar

   Compiler Flow


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
