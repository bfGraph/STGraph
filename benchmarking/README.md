# Benchmarking

The benchmarking is done in two categories. One for static-temporal graphs i.e. graphs with a static structure and varying node features. The second category is for dynamic graphs, where structure and features vary with time. For the static case we ahve considered the node classification task and for the dynamic case the link prediction task was used.

1. The `static-bench.sh` uses the WikiMaths and WindmillOutput datasets.
2. The `dynamic-bench.sh` uses the sx-mathoverflow and wiki-talk-temporal (pruned at 2Million temporal edges)

## Building the dataset

From the benchmarking/ directory run the following command. This will build the datasets in the necessary folders. It will also perform necessary preprocessing for the dynamic datasets. 

```
chmod u+x build-dataset.sh && ./build-dataset.sh
```

**Note:** Around 100 MB of data will be downloaded and the preprocessing could take around 5 minutes.