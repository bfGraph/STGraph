STGraph Changelog
=================

Welcome to the STGraph Changelog!

Here, you'll find a detailed record of the evolution of STGraph. The changelog provides a chronological 
overview of updates, enhancements, bug fixes, and new features introduced with each release.

What's new in v1.0.0?
---------------------

**Release Date**: 4th Nov, 2023

Announcing the official v1 release of STGraph, built on top of Seastar. STGraph is streamlined to have more 
reliable dependencies and supports execution out of the box. Additionally, we introduce support for TGNNs 
by introducing integrations with dynamic graph data structures.

Noting the immense amount of support provided by `@unnikrishnan-c <https://github.com/unnikrishnan-c>`_ and 
`@KevinJudeConcessao <https://github.com/KevinJudeConcessao>`_ in making these changes.

**New features**

* 🔥 Removed DGL Hack dependency
* 🤖 Making Seastar Backend Agnostic
* ➕ Added TGCN support for static-temporal graphs
* ➕ Added TGCN support for DTDGs
* ✨ Integrated PCSR and GPMA with STGraph for DTDG support
* ⚡ Optimized PCSR and GPMA integrations in STGraph
* 🕐 Benchmarked STGraph on 5 static-temporal and 5 DTDGs

**Contributors** : `@unnikrishnan-c <https://github.com/unnikrishnan-c>`_, `@KevinJudeConcessao <https://github.com/KevinJudeConcessao>`_, 
`@nithinmanoj10 <https://github.com/nithinmanoj10>`_, `@JoelMathewC <https://github.com/JoelMathewC>`_

View the `GitHub Changelog <https://github.com/bfGraph/STGraph/releases/tag/v1.0.0>`_

All Releases
------------

.. toctree::
   :maxdepth: 1
   :glob:

   1_0_0