=======
run_exp
=======

This is where the entire program starts from, when you run it on Google
Colab or any other platform. 

.. Note:: This is not where Seastar starts from

create_exp_list_sample0()
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: create_exp_list_sample0 (args)

    According to the different GNN models passed as input through the command line,
    ``create_exp_list_sample0`` returns a list (*exp_list*) of GNNExp class objects.

   :param args: Contains the arguments passed through the command line. Arguments are models, systems, gpu and num_epochs. 
    eg: ``args = Namespace(gpu=0, models=['gcn'], num_epochs=200, systems=['dgl', 'seastar'])``
   :rtype: List of GNNExp class objects. These objects can be from the following
    classes: GATExp, GCNExp, APPNPExp and RGCNExp

main()
^^^^^^

.. function:: main (args)

   Format the exception with a traceback.

   :param args: Contains the arguments passed through the command line. Arguments are models, systems, gpu and num_epochs. 
    eg: ``args = Namespace(gpu=0, models=['gcn'], num_epochs=200, systems=['dgl', 'seastar'])``
   :rtype: None

if __name__ == __main__:
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    parser = argparse.ArgumentParser(description='Experiment')

Within this if-block, the arguments which are passed through the
command line are parsed and stored inside an *argparse.ArgumentParser*
object.

::

    parser.add_argument("--models", nargs='+', default='gat', help="which models to run. Example usage: --models gat gcn")
    parser.add_argument("--systems", nargs='+', default='dgl', help="which models to run. Example usage: --systems dgl seastar pyg")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of training epochs")

- --models: Refer to the GNN models which are to be used for training eg: GCN, GAT, etc.
- --systems: Refer to the frameworks which are used for implementing these models eg: DGL, PyG and Seastar
- --gpu: To whether use a gpu or not
- --num_epochs: The number of epochs required for training

::

    args = parser.parse_args()
    args.models = [args.models] if not isinstance(args.models, list) else args.models
    args.systems = [args.systems] if not isinstance(args.systems, list) else args.systems

Parses the input command and stores the arguments in *args* . Also makes sure that
*args.models* and *args.systems* are list datatype.

This is what args would look like for the following command line input::

    >> ./run_exp.py --models gcn --systems dgl seastar
    args = Namespace(gpu=0, models=['gcn'], num_epochs=200, systems=['dgl', 'seastar'])


