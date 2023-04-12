===
exp
===

``exp`` (experiments) directory contains code for different GNN models like ``gcn``, ``gat``, ``rgcn`` and ``appnp`` with 
which the datasets are trained on. Each of the model stated above is implemented using ``dgl``, ``pyg`` and ``seastar``.

The ``result`` directory has a csv file *final_result.csv* which contains the results of the experiment/training 
conducted by each GNN model on the datasets.

.. toctree::
   :maxdepth: 1

   exp/appnp/APPNP
   exp/gat/GAT
   exp/gcn/GCN
   exp/rgcn/RGCN
   exp/run_exp

.. Attention:: Need to figure out ``nb_access`` and ``dataset`` directory
