Installation Guide
##################

Welcome to the STGraph installation guide that will lead you through the installation process ensuring a smooth setup
so you can start training TGNN models in your local machine.

**Currently STGraph is available for 🐍 Python 3.8 only**

.. note::

    It is always recommended to install STGraph and it's dependencies inside a dedicated python
    virtual environment with Python version set to 3.8

Installation for STGraph Package Users
======================================

This guide is tailored for users of the STGraph package, designed for constructing GNN and TGNN models. 
We recommend creating a new virtual environment with Python version 3.8 and installing stgraph inside that 
dedicated environment.

**Installing STGraph from PyPI**

.. code-block:: bash

    pip install stgraph

**Installing PyTorch**

In addition, STGraph relies on PyTorch. Ensure it is installed in your virtual environment with the following command

.. code-block:: bash

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Upon completion of the above steps, you have successfully installed STGraph. Proceed to write and 
train your first GNN and TGNN models.

Installation for STGraph Package Developers
===========================================

This guide is intended for those interested in developing and contributing to STGraph.

**Download source files from GitHub**

.. code-block:: bash

    git clone https://github.com/bfGraph/STGraph.git
    cd STGraph

**Create a dedicated virtual environment**

Inside the STGraph directory create and activate a dedicated virtual environment named dev-stgraph 
with Python version 3.8

.. code-block:: bash

    python3.8 -m venv dev-stgraph
    source dev-stgraph/bin/activate

**Install STGraph in editable mode**

Make sure to install the STGraph package in editable mode to ease your development process.

.. code-block:: bash

    pip install -e .[dev]
    pip list

**Install PyTorch**

Ensure to install PyTorch as well for development

.. code-block:: bash

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

With this you have successfully installed STGraph locally to make development changes and contribute to the project. 
Head out to our `Pull Requests <https://github.com/bfGraph/STGraph/pulls>`_ page and get started with your first contribution. 