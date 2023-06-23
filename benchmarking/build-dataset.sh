#! /bin/bash

cd dataset/static-temporal
wget https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json
wget https://graphmining.ai/temporal_datasets/windmill_output.json

cd ../dynamic-temporal
wget http://snap.stanford.edu/data/sx-mathoverflow.txt.gz && gzip -d sx-mathoverflow.txt.gz
wget http://snap.stanford.edu/data/wiki-talk-temporal.txt.gz && gzip -d wiki-talk-temporal.txt.gz

for i in 1.0
do
    python3 ../preprocessing/preprocess_temporal_data.py --dataset wiki-talk-temporal --base 1000000 --cutoff-time 2000000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-mathoverflow --base 250000 --percent-change $i
done

cd ../../


