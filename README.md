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

### Installation for STGraph Benchmarking

This guide is tailored for benchmarking of the STGraph. We recommend creating a new virtual environment with Python version `3.10` and installing `stgraph` inside that dedicated environment.

**Install the python packages**
```bash
python3.10 -m pip install -r requirements.txt
```

**Installing PyTorch, PyG and PyG-T**

In addition, STGraph relies on PyTorch. Ensure it is installed in your virtual environment with the following command

```bash
python3.10 -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
```bash
python3.10 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

```bash
python3.10 -m pip install torch_geometric
```

```bash
python3.10 -m pip install torch-geometric-temporal
```

### Building STGraph

Make sure to go back to the root directory and run the following to build and install STGraph

```
 python3.10 -m build && pip uninstall stgraph -y && pip install dist/stgraph-1.0.0-py3-none-any.whl
```

## Authors

| Author                            | Bio                                                                   |
| --------------------------------- | --------------------------------------------------------------------- |
| `Joel Mathew Cherian`             | Computer Science Student at National Institute of Technology Calicut  |
| `Nithin Puthalath Manoj`          | Computer Science Student at National Institute of Technology Calicut  |
| `Dr. Unnikrishnan Cheramangalath` | Assistant Professor in CSED at Indian Institue of Technology Palakkad |
| `Kevin Jude`                      | Ph.D. in CSED at Indian Institue of Technology Palakkad               |

## References

| Author(s)                                                                                                                                                                         | Title                                                                                                    | Link(s)                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `Wu, Yidi and Ma, Kaihao and Cai, Zhenkun and Jin, Tatiana and Li, Boyang and Zheng, Chenguang and Cheng, James and Yu, Fan`                                                      | `STGraph: vertex-centric programming for graph neural networks, 2021`                                    | [paper](https://doi.org/10.1145/3447786.3456247), [code](https://zenodo.org/record/4988602)                                      |
| `Wheatman, Brian and Xu, Helen`                                                                                                                                                   | `Packed Compressed Sparse Row: A Dynamic Graph Representation, 2018`                                     | [paper](https://ieeexplore.ieee.org/abstract/document/8547566), [code](https://github.com/wheatman/Packed-Compressed-Sparse-Row) |
| `Sha, Mo and Li, Yuchen and He, Bingsheng and Tan, Kian-Lee`                                                                                                                      | `Accelerating Dynamic Graph Analytics on GPUs, 2017`                                                     | [paper](http://www.vldb.org/pvldb/vol11/p107-sha.pdf), [code](https://github.com/desert0616/gpma_demo)                           |
| `Benedek Rozemberczki, Paul Scherer, Yixuan He, George Panagopoulos, Alexander Riedel, Maria Astefanoaei, Oliver Kiss, Ferenc Beres, Guzmán López, Nicolas Collignon, Rik Sarkar` | `PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models, 2021` | [paper](https://arxiv.org/pdf/2104.07788.pdf), [code](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)         |
