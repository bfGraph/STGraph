#! /bin/bash

cd dataset/dynamic-temporal
gdown --fuzzy "https://drive.google.com/file/d/1_oKkXG_3aIA5r-Jsnx5GY4birAO0bF5U/view?usp=sharing"
gdown --fuzzy "https://drive.google.com/file/d/1ir2-csd2FNk4JTpYnpXveVNSCgJemJPk/view?usp=sharing"
gdown --fuzzy https://drive.google.com/file/d/16NZG09NjHZuOF8IP34SMvNUm_24C1822/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1WX8HH1rWXVH5iV_OFyrhndqesnfmgSBd/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1-AQBv1Z_XWqTfHv6jF85Vq-oVjJMlfIm/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1-0-SQB-ZVa6WrRkU7yjX6prS7yuVDbYD/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1-7_R8reuthVzB2hIy_G5bjwLwYvWxqzT/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1-3y0lBwHnSIU3m83JfdKYRei7uWE48tt/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1dlgyxnNOXmbnI3bHTWD_5CtsUhKP0xbO/view?usp=sharing

for i in 2.0 4.0 5.0 6.0 8.0 10.0
do
    python3 ../preprocessing/preprocess_temporal_data.py --dataset wiki-talk-temporal --base 1000000 --cutoff-time 2000000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-mathoverflow --base 250000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-askubuntu --base 480000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-superuser --base 720000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset sx-stackoverflow --base 1000000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset reddit-title --base 420000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset reddit-body --base 420000 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset bitcoin-otc --base 17500 --percent-change $i
    python3 ../preprocessing/preprocess_temporal_data.py --dataset email-eu-core-temporal --base 160000 --percent-change $i
done

cd ../../