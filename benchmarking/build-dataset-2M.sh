#! /bin/bash

cd dataset/dynamic-temporal

for i in 2.0 4.0 5.0 6.0 8.0 10.0
do
    python3 ../preprocessing/preprocess_temporal_data.py --dataset wiki-talk-temporal --base 1000000 --cutoff-time 2000000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-stackoverflow --base 1000000 --cutoff-time 2000000 --percent-change $i
done

cd ../../