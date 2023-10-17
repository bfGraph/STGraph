Installation Guide
##################

Welcome to the STGraph installation guide that will lead you through the installation process, including prerequisites, ensuring a smooth setup
so you can start training TGNN models in your local machine.

Installing STGraph
==================

Begin by downloading STGraph source files from GitHub to your local machine.

.. code-block:: bash

    git clone https://github.com/bfGraph/STGraph.git
    cd STGraph

Setting Up a Virtual Environment
================================

We strongly recommend creating a dedicated Python virtual environment. You can choose either conda or venv.

**Using conda**

.. code-block:: bash

    conda create --name stgraph
    conda activate stgraph

Installing Required Python Packages
===================================

To install the necessary Python packages for STGraph, use the following command. Remember to activate the ``stgraph`` virtual environment before proceeding.

.. code-block:: bash

    pip install -r requirements.txt

.. note::

    You may encounter some errors or warnings during this process. You can safely ignore them for now.

PyTorch and PyG-T Installation
==============================

It's advisable to install PyTorch, torchvision, torchaudio, and PyTorch Geometric Temporal separately. 

.. code-block:: bash

    pip install torch torchvision torchaudio
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    pip install torch-geometric-temporal

STGraph Graph Packages Installation
===================================

Before proceeding further, it's essential to build the shared object (.so) file for the CUDA/C++ extension responsible for certain dynamic and static graph representations in STGraph.

.. code-block:: bash

    cd stgraph/graph/
    ./build_static.sh
    ./build_dynamic.sh gpma pcsr

After a successful build, you can proceed with the remaining installation steps.

Completing STGraph Installation
===============================

To finalize the installation of the STGraph package, execute these commands.

.. code-block:: bash

    cd ../..
    python3 -m build && pip uninstall stgraph -y && pip install dist/stgraph-1.0.0-py3-none-any.whl

After running these commands, confirm the correct installation of STGraph by executing

.. code-block:: bash

    pip show stgraph

Following these instructions will result in a successful STGraph installation, complete with all the necessary packages.

CUDA Python
===========

STGraph leverages CUDA Python, offering low-level interfaces for CUDA host APIs within Python. Ensure your system meets the following requirements before proceeding.

.. note::

    STGraph is tested with Python CUDA version 11.7 and above, and this guide is based on Python CUDA version 12.1.0. Installation is also possible for versions 11.7 and above.

**System Requirements**

1. **Driver**: Linux (450.80.02 or later) / Windows (456.38 or later)
2. **CUDA Toolkit**: Versions 12.0 to 12.1
3. **Python**: Versions 3.8 to 3.11

Installation via PyPI
---------------------

Install CUDA Python via PyPI with the following command

.. code-block:: bash

    pip install cuda-python

Verification
------------

To verify the successful installation of Python CUDA, run the provided Python script located within the STGraph directory.

.. code-block:: bash

    cd stgraph/compiler/code_gen/
    python3 cuda_check.py

A successful installation will yield output similar to the following, with specifics dependent on your GPU

.. code-block:: 

           Device Property   Value
    ───────────────────────────────────────────────────
         Number of Devices   1
                      Name   NVIDIA GeForce MX350
        Compute Capability   6.1
      Multiprocessor Count   5
        Concurrent Threads   10240
                 GPU Clock   1468.0 MHz
              Memory Clock   3504.0 MHz
              Total Memory   2047.875 MiB
               Free Memory   1641.7216796875 MiB

With this, you've completed the STGraph installation and verified CUDA Python's presence on your system.

Running STGraph
===============

To confirm the successful installation, let's proceed with running STGraph by training a T-GCN model on the WikiMaths dataset.

.. code-block:: bash

    cd ../../..
    cd benchmarking/
    chmod u+x verify.sh
    ./verify.sh

Executing the above command will initiate the dataset download and model training. A table will display the time taken, MSE, and memory consumption for each epoch. Congratulations, STGraph is running as expected.

In some cases, you may encounter out-of-memory (OOM) issues. This can occur when your GPU lacks the necessary memory for this dataset. However, you can still validate STGraph's functionality by inspecting the CUDA code generated for this T-GCN model, found in the following directory

.. code-block:: bash

    cd static-temporal-tgcn/stgraph/

Within this directory, you'll find the CUDA code in ``egl_kernel.cu`` and the PTX file in ``egl_kernel.ptx``.

If you encounter any errors while attempting to train the T-GCN model, kindly raise an issue, and our team will promptly assist you.