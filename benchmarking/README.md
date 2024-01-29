# Benchmarking

The benchmarking is done in two categories. One for static-temporal graphs i.e. graphs with a static structure and varying node features. The second category is for dynamic graphs, where structure and features vary with time. For the static case we have considered the node classification task and for the dynamic case the link prediction task was used.

1. The `static-bench.sh` 
2. The `dynamic-bench.sh` 


## Building the dataset

From the benchmarking/ directory run the following command. This will build the datasets in the necessary folders. It will also perform necessary preprocessing for the dynamic datasets. 

```
chmod u+x build-dataset.sh && ./build-dataset.sh
```

**Note:** Around 3 GB of data will be downloaded and the preprocessing could take around 10-15 minutes.

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
