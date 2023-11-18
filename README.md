<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/bfGraph/STGraph/blob/main/assets/STGraph%20Banner%20%E2%80%93%20Dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/bfGraph/STGraph/blob/main/assets/STGraph%20Banner%20New.png?raw=true">
  <img alt="STGraph Banner" src="https://github.com/bfGraph/STGraph/blob/main/assets/STGraph%20Banner%20New.png?raw=true">  
</picture>


[![Documentation Status](https://readthedocs.org/projects/stgraph/badge/?version=latest)](https://stgraph.readthedocs.io/en/latest/?badge=latest)
[![TGL Workshop - @ NeurIPS'23](https://img.shields.io/badge/TGL_Workshop-%40_NeurIPS'23-6d4a8f)](https://neurips.cc/virtual/2023/76335)
[![PyPI - 1.0.0](https://img.shields.io/static/v1?label=PyPI&message=1.0.0&color=%23ffdf76&logo=Python)](https://pypi.org/project/stgraph/)

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
    .
    <a href="https://openreview.net/forum?id=8PRRNv81qB">Paper</a>
  </p>
</div>


# 🌟 STGraph

STGraph is a framework designed for deep-learning practitioners to write and train Graph Neural Networks (GNNs) and Temporal Graph Neural Networks (TGNNs). It is built on top of _Seastar_ and utilizes the vertex-centric approach to produce highly efficient fused GPU kernels for forward and backward passes. It achieves better usability, faster computation time and consumes less memory than state-of-the-art graph deep-learning systems like DGL, PyG and PyG-T.

_NOTE: If the contents of this repository are used for research work, kindly cite the paper linked above._

## Why STGraph

![Seastar GCN Formula](https://github.com/bfGraph/STGraph/blob/main/assets/Seastar%20GCN%20Formula.png?raw=true)



The primary goal of _Seastar_ is more natural GNN programming so that the users learning curve is flattened. Our key observation lies in recognizing that the equation governing a GCN layer, as shown above, takes the form of vertex-centric computation and can be implemented succinctly with only one line of code. Moreover, we can see a clear correspondence between the GNN formulas and the vertex-centric implementations. The benefit is two-fold: users can effortlessly implement GNN models, while simultaneously understanding these models by inspecting their direct implementations.

The _Seastar_ system outperforms state-of-the-art GNN frameworks but lacks support for TGNNs. STGraph bridges that gap and enables users to to develop TGNN models through a vertex-centric approach. STGraph has shown to be significantly faster and more memory efficient that state-of-the-art frameworks like PyG-T for training TGNN models.

## Getting Started

### Installation for STGraph Package Users

This guide is tailored for users of the STGraph package, designed for constructing Graph Neural Network (GNN) and Temporal Graph Neural Network (TGNN) models.

**Setting Up a Virtual Environment**

We recommend creating a new virtual environment with Python version `3.8`. Here, we create a dedicated virtual environment named `stgraph` using the `conda` package manager.

```bash
conda create --name stgraph python=3.8
conda activate stgraph
```

**Installing STGraph from PyPI**

```bash
pip install stgraph
```

**Installing PyTorch**

In addition, STGraph relies on PyTorch. Ensure it is installed in your virtual environment with the following command

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Upon completion of the above steps, you have successfully installed STGraph. Proceed to write and train your first GNN model by referring to the provided tutorial.

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
