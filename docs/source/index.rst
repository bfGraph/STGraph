.. STGraph documentation master file, created by
   sphinx-quickstart on Tue Oct 17 14:51:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/stgraph_docs_banner.png
   :alt: STGraph Docs Banner

🌟 STGraph
==========

STGraph is a cutting-edge deep learning framework, built on top of Seastar, designed
to easily write and train GNNs and TGNNs on real life graph datasets.

With a focus on TGNNs to capture both inter-node interactions and node-specific interactions in dynamic graphs,
STGraph extends the powerful Seastar vertex-centric programming model to support TGNNs, enabling efficient training on GPUs.
Unlike existing solutions that store dynamic temporal graphs as separate snapshots, leading to high memory overhead, 
STGraph dynamically constructs snapshots on demand during training. This innovative approach leverages dynamic graph 
data structures, efficiently adapting to temporal updates. 

Explore the STGraph documentation and tutorials to get started with writing and training your on GNNs and TGNNs.

.. toctree::
   :maxdepth: 1
   :caption: Install STGraph
   :glob:

   install/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :glob:

   tutorials/gcn_cora

.. toctree::
   :maxdepth: 1
   :caption: Package Reference
   :glob:

   package_reference/stgraph.dataset
   package_reference/stgraph.compiler
   package_reference/stgraph.graph
   package_reference/stgraph.benchmark_tools
   package_reference/stgraph.nn

.. toctree::
   :maxdepth: 1
   :caption: Changelog
   :glob:

   changelogs/index

