<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/bfGraph/Seastar/assets/64126131/dfa8aa12-a047-43d7-a938-67e1e52f081b">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/bfGraph/Seastar/assets/64126131/86b75d51-8701-47b3-88fc-fbc0f070693b">
  <img alt="STGraph Banner" src="https://github.com/bfGraph/STGraph/assets/64126131/36a8bb80-25a9-4684-b604-7a5efbea12d4">  
</picture>


[![Documentation Status](https://readthedocs.org/projects/stgraph/badge/?version=latest)](https://stgraph.readthedocs.io/en/latest/?badge=latest)

<div align="center">
  <p align="center">
    <a href="https://stgraph.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/bfGraph/STGraph/blob/main/INSTALLATION.md">Installation guide</a>
    ·
    <a href="https://github.com/bfGraph/STGraph/issues">Report Bug</a>
    ·
    <a href="https://github.com/bfGraph/STGraph/discussions">View Discussions</a>
  </p>
</div>


# 🌟 STGraph

STGraph is a cutting-edge system designed to enhance the performance of graph neural network (GNN) and Temporal-GNN training on GPUs. Utilize the vertex-centric approach to programming a GNN to produce highly efficient fused GPU kernels for forward and backward passes. It achieves better usability, faster computation time and consumes less memory than state-of-the-art GNN systems like DGL and PyG.

## Why STGraph

![STGraph GCN Formula](https://github.com/bfGraph/STGraph/blob/main/assets/STGraph%20GCN%20Formula.png?raw=true)

STGraphs' objective is more natural GNN programming so that users’ learning curve is flattened. Our key observation is that the equation for a GCN layer shown above takes the form of vertex-centric computation, i.e., it computes the features of a center vertex 𝑣 by aggregating the features of its neighbours.

The key step of GCN can be implemented succinctly with only one line of code. More importantly, we can see a clear correspondence between the GNN formulas and the vertex-centric implementations. The benefit is bi-directional: users can implement a GNN model easily, and users can learn the GNN model by directly checking its implementation.

## Getting Started

### Pre-requisites

Install the python packages by running the following. It is recommended that you create a virtual environment before installing the packages.

**Setup a new virtual environment**
```
conda create --name stgraph
conda activate stgraph
```

**Install the python packages**
```
pip install -r requirements.txt
```

STGraph requires CUDA Version `11.7` or above to run. Versions below that may or may not run, depending on any changes within CUDA.

### Installation

You can git clone STGraph into your workspace by running the following command

```
git clone https://github.com/bfGraph/STGraph.git
cd STGraph
```

## Running your first STGraph Program

In this is quick mini tutorial, we will show you how to train a simple GCN model on the Cora dataset. After installing STGraph and entering the STGraph directory, enter the following commands to reach the GCN `benchmarking` folder

```
cd benchmarking/gcn/stgraph
```

Run the `train.py`, with 100 epochs and specify the dataset name. For this example, we shall use Cora

```
python3 train.py --num_epochs 100 --dataset cora
```

You should get an output like this. The initial prints are truncated.

```
.
.
.
Epoch 00090 | Time(s) 0.0048 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00091 | Time(s) 0.0024 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00092 | Time(s) 0.0029 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00093 | Time(s) 0.0029 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00094 | Time(s) 0.0027 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00095 | Time(s) 0.0030 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00096 | Time(s) 0.0024 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00097 | Time(s) 0.0022 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00098 | Time(s) 0.0022 | train_acc 0.303791 | Used_Memory 32.975098 mb
Epoch 00099 | Time(s) 0.0036 | train_acc 0.303791 | Used_Memory 32.975098 mb

^^^0.032202^^^0.003098
```

If you don't get this output and have followed every single step in the setting up and installation section, please raise an issue we will look into it.

## How to build STGraph

This is for users who want to make changes to the STGraph codebase and get it build each time. Follow the steps mentioned to properly build STGraph.

### Compiling the CUDA code

The following steps need to be done if you made any changes to any CUDA files within the `stgraph/graph` directory for each graph representation.

STGraph supports training dynamic and static graphs. To handle all the graph representations logic, it is written as a PyBind11 module over a CUDA file. As of now the following CUDA code for different graph representations exists

1. `csr.cu`
2. `pcsr.cu`
3. `gpma.cu`

To compile the `[name].cu` file, run the following command

```
/usr/local/cuda-11.7/bin/nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o [name].so [name].cu
```
This would generate the [name].so shared object file, that is used in the STGraph module. 

### Building STGraph

Make sure to go back to the root directory and run the following to build and install STGraph

```
 python3 -m build && pip uninstall stgraph -y && pip install dist/stgraph-1.0.0-py3-none-any.whl
```

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests, issues, etc to us.

## How to contribute to Documentation

We follow the PEP-8 format. [Black](https://pypi.org/project/black/) is used as the formatter and [pycodestyle](https://pypi.org/project/pycodestyle/) as the linter. The linter is is configure to work properly with black (set line length to 88)

Tutorial for Python Docstrings can be found [here](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)

```
sphinx-apidoc -o docs/developers_guide/developer_manual/package_reference/ python/stgraph/ -f
cd docs/
make clean
make html
```
## Authors

| Author                            | Bio                                                                   |
| --------------------------------- | --------------------------------------------------------------------- |
| `Joel Mathew Cherian`             | Computer Science Student at National Institute of Technology Calicut  |
| `Nithin Puthalath Manoj`          | Computer Science Student at National Institute of Technology Calicut  |
| `Dr. Unnikrishnan Cheramangalath` | Assistant Professor in CSED at Indian Institue of Technology Palakkad |
| `Kevin Jude`                      | Ph.D. in CSED at Indian Institue of Technology Palakkad               |

## Attributions

| Author(s)                                                                                                                                                                         | Title                                                                                                    | Link(s)                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `Wu, Yidi and Ma, Kaihao and Cai, Zhenkun and Jin, Tatiana and Li, Boyang and Zheng, Chenguang and Cheng, James and Yu, Fan`                                                      | `STGraph: vertex-centric programming for graph neural networks, 2021`                                    | [paper](https://doi.org/10.1145/3447786.3456247), [code](https://zenodo.org/record/4988602)                                      |
| `Wheatman, Brian and Xu, Helen`                                                                                                                                                   | `Packed Compressed Sparse Row: A Dynamic Graph Representation, 2018`                                     | [paper](https://ieeexplore.ieee.org/abstract/document/8547566), [code](https://github.com/wheatman/Packed-Compressed-Sparse-Row) |
| `Sha, Mo and Li, Yuchen and He, Bingsheng and Tan, Kian-Lee`                                                                                                                      | `Accelerating Dynamic Graph Analytics on GPUs, 2017`                                                     | [paper](http://www.vldb.org/pvldb/vol11/p107-sha.pdf), [code](https://github.com/desert0616/gpma_demo)                           |
| `Benedek Rozemberczki, Paul Scherer, Yixuan He, George Panagopoulos, Alexander Riedel, Maria Astefanoaei, Oliver Kiss, Ferenc Beres, Guzmán López, Nicolas Collignon, Rik Sarkar` | `PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models, 2021` | [paper](https://arxiv.org/pdf/2104.07788.pdf), [code](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)         |
