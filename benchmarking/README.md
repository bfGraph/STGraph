# Benchmarking

The benchmarking is done in two categories. One for static-temporal graphs i.e. graphs with a static structure and varying node features. The second category is for dynamic graphs, where structure and features vary with time. For the static case we have considered the node classification task and for the dynamic case the link prediction task was used.

1. The `static-bench.sh` uses the WikiMaths and WindmillOutput datasets.
2. The `dynamic-bench.sh` uses the sx-mathoverflow and wiki-talk-temporal (pruned at 2Million temporal edges)

## Setting up STGraph

1. From the main project directory run the collowing commands
```
pip install -r requirements.txt
```

2. Install python venv
```
sudo apt-get install python3.10-venv
```

3. Traverse to the `stgraph/graph/static` folder
```
nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o csr.so csr.cu
```

4. Traverse to the `stgraph/graph/dynamic/pcsr` folder
```
nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o pcsr.so pcsr.cu
```

5. Traverse to the `stgraph/graph/dynamic/gpma` folder
```
nvcc $(python3 -m pybind11 --includes) -shared -rdc=true --compiler-options '-fPIC' -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -o gpma.so gpma.cu
```

6. From the main project directory run the following
```
python3 -m build && pip uninstall stgraph -y && pip install dist/stgraph-1.0.0-py3-none-any.whl
```

**Note:** For benchmarking we will need to install PyG-T, the following commands should do the necessary
```
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric-temporal
```


## Building the dataset

From the benchmarking/ directory run the following command. This will build the datasets in the necessary folders. It will also perform necessary preprocessing for the dynamic datasets. 

```
chmod u+x build-dataset.sh && ./build-dataset.sh
```

**Note:** Around 100 MB of data will be downloaded and the preprocessing could take around 5 minutes.

## Running the benchmark scripts

The benchmark scripts will execute a series of scripts with a variation of parameters. The logs of all these can be found in the `results/` folder at the end of the benchmarking.

1. To run the static-temporal graphs tests

```
chmod u+x static-bench.sh && ./static-bench.sh
```

2. To run the dynamic graph tests

```
chmod u+x dynamic-bench.sh && ./dynamic-bench.sh
```

**Note:** Each of these tests could take upto an hour or more.

# Benchmarking Extension

The following datasets were added for the following benchmarking categories. 

### Spatio-Temporal Datasets

1. Hungary Chickenpox
2. PedalMe
3. MonteVideoBus

### Dynamic Datasets

1. sx-mathoverflow
2. sx-stackoverflow
3. sx-askubuntu
4. soc-reddit-hyperlink-title
5. soc-reddit-hyperlink-body
6. wiki-talk-temporal
7. email-eu-core-temporal
8. sx-superuser
9. bitcoin OTC

Make sure to download the new datasets first by running the `ext-build-dataset.sh` file
```bash
chmod u+x ext-build-dataset.sh && ./ext-build-dataset.sh
```

## Running the new benchmarking scripts

The benchmark scripts will execute a series of scripts with a variation of parameters. The logs of all these can be found in the `results/` folder at the end of the benchmarking.

1. To run the new static-temporal graphs tests

```
chmod u+x ext-static-bench.sh && ./ext-static-bench.sh
```

2. To run the new dynamic graphs tests

```
chmod u+x ext-dynamic-bench.sh && ./ext-dynamic-bench.sh
```
