#! /bin/bash
# Script to download the following new datasets

# Spatio-temporal Datasets
# 1. Hungary Chicken Pox
# 2. PedalMe
# 3. METRLA
# 4. Montevideo Bus

cd dataset/static-temporal
wget https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/HungaryCP.json
wget https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/pedalme.json
wget https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/METRLA.json
wget https://raw.githubusercontent.com/bfGraph/Seastar-Datasets/main/montevideobus.json

# TODO: Download links for new dynamic datasets

cd ../../