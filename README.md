![Seastar Banner](https://github.com/bfGraph/Seastar/blob/main/assets/Seastar%20Banner.png?raw=true)
[![Documentation Status](https://readthedocs.org/projects/seastar/badge/?version=latest)](https://seastar.readthedocs.io/en/latest/?badge=latest)

# Seastar

Seastar is a cutting-edge system designed to enhance the performance of graph neural network (GNN) and Temporal-GNN training on GPUs. Utilize the vertex-centric approach to programming a GNN to produce highly efficient fused GPU kernels for forward and backward passes. It achieves better usability, faster computation time and consumes less memory than state-of-the-art GNN systems like DGL and PyG.

## Why Seastar

Show an image of the GCN layer formula, similar to the Seastar GCN version.

Seastars' objective is more natural GNN programming so that users’ learning curve is flattened. Our key observation is that the equation for a GCN layer shown above takes the form of vertex-centric computation, i.e., it computes the features of a center vertex 𝑣 by aggregating the features of its neighbours.

The key step of GCN can be implemented succinctly with only one line of code. More importantly, we can see a clear correspondence between the GNN formulas and the vertex-centric implementations. The benefit is bi-directional: users can implement a GNN model easily, and users can learn the GNN model by directly checking its implementation.

## Getting Started

### Pre-requisites

Install the python packages by running the following. It is recommended that you create a virtual environment before installing the packages.

**Setup a new virtual environment**
```
conda create --name seastar
conda activate seastar
```

**Install the python packages**
```
pip install -r requirements.txt
```

Seastar requires CUDA Version `11.7` or above to run. Versions below that may or may not run, depending on any changes within CUDA.

### Installation

You can git clone Seastar into your workspace by running the following command

```
git clone https://github.com/bfGraph/Seastar.git
cd Seastar
```

## How to contribute to Documentation

We follow the PEP-8 format. [Black](https://pypi.org/project/black/) is used as the formatter and [pycodestyle](https://pypi.org/project/pycodestyle/) as the linter. The linter is is configure to work properly with black (set line length to 88)

Tutorial for Python Docstrings can be found [here](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)

```
sphinx-apidoc -o docs/developers_guide/developer_manual/package_reference/ python/seastar/ -f
cd docs/
make clean
make html
```


## Attributions

Yidi WU. (2021). 21 April 2021Seastar: vertex-centric programming for graph neural networks. https://doi.org/10.1145/3447786.3456247
