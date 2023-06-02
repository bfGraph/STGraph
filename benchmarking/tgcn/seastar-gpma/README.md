# Benchmark TGCN with Seastar GPMA

Using GPMAGraph to benchmark TGCN on the Foorah Dataset

## Running the benchmark

```bash
python3 dynamic_bench.py --max_feat_size 16 --max_num_nodes 55000 --num_epochs 10
```

you should get an output like this

```
                  Benchmarking T-GCN with GPMA

    Dataset Name ┃ Feat. Size ┃ Total Time ┃ copy_label_edges()
━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━
  foorah_large_8 │ 8          │ 1.1659     │ 0
 foorah_large_16 │ 16         │ 1.2278     │ 0
```