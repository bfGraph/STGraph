Training a GCN on the Cora dataset
==================================

Graph Neural Networks (GNNs) are specially designed to understand and learn from data organized in graphs, 
making them incredibly versatile and powerful. Graph Convolutional Networks (GCNs) is a widely adopted
model which makes use of both node features and local connections.

In this introductory tutorial, you will be able to 

1. Build a GCN model using STGraph's neural network layers.
2. Load the Cora dataset provided by STGraph.
3. Train and evaluate the GCN model for node classification task on the GPU.

The task at hand
----------------

The Cora dataset is a widely used citation network for benchmarking graph-based machine learning algorithms.
It comprises and captures the relationship between 2708 scientific publications classified into one of seven classes, 
where nodes represent individual papers, and edges denote citation links between them. The network comprises of 
5,429 connections. Each publication in the dataset is characterized by a binary word vector (0 or 1), 
signifying the non-existence or existence of the respective word from a dictionary of 1,433 unique words.

Our task is to train a GCN model on the Cora dataset and predict the topic of a publication (node) by considering 
the neighboring node information and the overall graph structure. Or in other words, Node Classification.

.. figure:: ../_static/Images/tutorials/CoraBalloons.png
   :alt: CoraBalloons
   :align: center
   :width: 400

   Cora Dataset Visualized [1]

Writing the GCN model
---------------------

Let's start by building our GCN model within a file named ``model.py``. First, import all the required modules. We will use PyTorch as our backend framework,
along with the ``GraphConv`` layer from STGraph, which is designed for the PyTorch backend.

.. code-block:: python

    import torch.nn as nn
    from stgraph.nn.pytorch.graph_conv import GraphConv

Our main component is the GCN class, which represents the Graph Convolutional Network we will train. Here’s the code to initialize the GCN object

.. code-block:: python

    class GCN(nn.Module):
        def __init__(
            self,
            graph: StaticGraph,
            n_input: int,
            n_hidden: int,
            n_output: int,
            n_layers: int,
            activation
        ):

            super(GCN, self).__init__()

            self.graph = graph
            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(n_input, n_hidden, activation))
            
            for i in range(n_layers - 1):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation))
            
            self.layers.append(GraphConv(n_hidden, n_output, None))

First, let's review all the arguments passed to the initialization method

1. **graph**: This should be an STGraph graph object representing our graph dataset. In this case, the Cora dataset will be of type ``StaticGraph``.
2. **n_input**: This refers to the number of neurons in the input layer of our GCN.
3. **n_hidden**: This specifies the number of neurons in each hidden layer. We assume all hidden layers have the same number of neurons.
4. **n_output**: This is the number of neurons in the output layer.
5. **n_layers**: This keeps track of the total number of non-input layers, including all hidden layers and the output layer.
6. **activation**: This is the element-wise activation function we will use for each layer.

We will initialize a list to hold all the layers of our GCN model. Using ``nn.ModuleList()`` allows for easier management of these layers. To this list,
we will append ``GraphConv`` layers for the input layer, all the hidden layers, and then the output layer. We specify the number of neurons present as input and
output as we propagate through each layer. Note that we use an element-wise activation function only for the input and hidden layers,
as the output layer typically does not use an activation function.

