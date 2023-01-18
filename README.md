# seastar-paper-version
This is a snapshot of Seastar when paper is submitted. It is not mature at all. It runs smoothly following the instructions listed below but users may encounter various bugs and unexpected behaviours when play around.

## Folder organization
python/: source code of seastar

exp/: experiments

## Getting Started
A linux workstation with at least one NVIDIA GPU equipped is required to run Seastar. We've tested on 1080 Ti, 2080 Ti and V100 with CUDA version 10.1.

### Setup anaconda environment
Anaconda creates a clean execution environment and simplifies pacakage management. 
1. Download and install [anaconda](https://docs.anaconda.com/anaconda/install/linux/).
2. Create a Python 3.7 (or higher) environment. "conda create -n seastar python=3.7".
3. Activate conda environment before installing any dependecies with "conda activate seastar".

### Install Dependencies
It may take **upto** a few hours on this section.

To run the performance comparison experiments, we need to install PyTorch, PyG, DGL and Seastar.

#### PyTorch
1. Run "conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch" (replace cudatoolkit=10.1 with the CUDA version with what is shown by running "nvidia-smi").

2. Verify installation with "python -c "import torch; print(torch.\_\_version\_\_)"". 1.6.0 should be printed on terminal.

#### PyTorch Geometric (PyG)
1. First verify scipy installation.

2. Follow the instruction of [PyG's installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install PyG. Note a version that supports PyTorch-1.6.0 should be chosen. Installation via Binaries is recommended.

3. Verify installation with "python -c "import torch_geometric as pyg; print(pyg.\_\_version\_\_)"". A version number will be printed on terminal.


#### DGL
The snapshot relies on **dgl-hack**, which a hacked version of DGL-v0.4.3. We added several customized kernels without touching the rest.

1. Install cmake first by running "conda install -c anaconda cmake". Verify cmake installation with "cmake --version". 
2. Run "git clone --recursive https://github.com/ydwu4/dgl-hack".
3. Build and install with "cd dgl-hack && mkdir build && cd build && cmake .. && cd .. && ./compile.sh". Tested with cmake 3.18.3 and gcc 7.3.0.
4. Verify installation with "python -c "import dgl; print(dgl.\_\_version\_\_)"".


#### Seastar
1. Run "git clone https://github.com/ydwu4/seastar-paper-version"
2. Install with "cd seastar-paper-version/python && python setup.py install"
3. Verify installation with "python -c "import seastar"". You should see a bunch of op registration message.

### Run experiments
This section can take a few hours to finish. You can significantly shorten the running time by not training on reddit dataset.

#### Prepare datasets
Following the readme.md in exp/dataset to preprare the datasets.

#### System Performance comparison
Run "python ./run_exp.py --models gat gcn appnp rgcn --systems dgl pyg seastar" to run models gat, gcn, appnp and rgcn on systems dgl, pyg and seastar respectively.

The exact command for each dataset, model and system to execute is printed on terminal. Users can directly invoke the corresponding command without using the run_exp.py script.

For each experiment, the peak memory usage (GB) and per-epoch time (s) are printed out. Users can compare those numbers with that are shown in Figure 8, 9 and Table 3, 4 of paper. The results will also be written in ./result/final_result.csv after the script finishes execution.
 
Note that seastar may dump egl_kernel.cu and egl_kernel.ptx in current folder for some models (later execution will override the previous files). You may inspect these generated kernels.

#### Kernel Micro-benchmarks
Run "chmod u+x run_nb_access_bench.sh" and "./run_nb_access_bench.sh" to execute the micro benchmark

We run the benchmark for both sorted and unsorted graph and print results for unsorted graph first then sorted graph. For each graph, there are 4 schemes: baseline, basic, FA + atomic, and FA + dynamic. So totally there will be 2 * 4 = 8 schemes. The corresponence between paper and the microbenchmark is shown as follows

1. basline -> baseline + unsorted;
2. Basic -> basic + unsorted;
3. FA + Unsorted -> FA + dynamic + Unsorted
4. FA + Sorting + Atomic -> FA + atomic + sorted
5. FA + Sorting + dynamic -> FA + dynamic + sorted

, where on arrow's left is the notation used in paper and the right is benchmark setting.

## Play around
### Changing model configurations
You could specify various model configurations in the constructor of the corresponding Exp class (defined in run_exp.py) by writing a different create_exp_list function or directly write a command by insepcting the trian.py for each model.

### Writing new models using vertex-centric programming model

Following the GCN, GAT, APPNP model implementation, users may try to implement exisiting models (e.g. can find model in DGL conv layer library). Note that this version of seastar requires users to extract graph structure form dgl graph mannuly using get_immutable_gidx(). We will clean the API in the official release. Users most likely will enconter various unexpected behaviors.
