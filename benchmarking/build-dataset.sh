#! /bin/bash

cd dataset/static-temporal
wget https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/wikivital_mathematics.json
wget --no-check-certificate https://graphmining.ai/temporal_datasets/windmill_output.json

# NOTE: These datasets are exposed via a personal google drive because as of June 2023 Stanford SNAP is down
# The only files on this drive are intended for the use of this benchmarking alone.
cd ../dynamic-temporal
# wget http://snap.stanford.edu/data/sx-mathoverflow.txt.gz && gzip -d sx-mathoverflow.txt.gz
# wget http://snap.stanford.edu/data/wiki-talk-temporal.txt.gz && gzip -d wiki-talk-temporal.txt.gz
gdown --fuzzy "https://drive.google.com/file/d/1_oKkXG_3aIA5r-Jsnx5GY4birAO0bF5U/view?usp=sharing"
gdown --fuzzy "https://drive.google.com/file/d/1ir2-csd2FNk4JTpYnpXveVNSCgJemJPk/view?usp=sharing"

for i in 2.0 4.0 5.0 6.0 8.0 10.0
do
    python3 ../preprocessing/preprocess_temporal_data.py --dataset wiki-talk-temporal --base 1000000 --cutoff-time 2000000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-mathoverflow --base 250000 --percent-change $i
done

cd ../../


