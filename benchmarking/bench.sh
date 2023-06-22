#! /bin/bash
# Naming scheme (pygt|seastar)_(dataset_name)_T(cutoff_time|whole)_B(backprop_every|whole)_F(feature_size)

# (Static Temporal PYG-T)
cd static-temporal-tgcn/pygt

# ---- (PYGT) WikiMaths ----
python3 train.py --dataset wiki --num-epochs 10 > ../../results/static-temporal/pygt_wikimaths_Twhole_Bwhole_F8.txt

for i in {8..80..8}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size $i --backprop-every 1000 > ../../results/static-temporal/pygt_wikimaths_Twhole_Bwhole_F$i.txt
done

for i in {2000..17000..2000}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size 8 --backprop-every $i  > ../../results/static-temporal/pygt_wikimaths_Twhole_B$i\_F8.txt
done

# ---- (PYGT) Windmill----
python3 train.py --dataset windmill --num-epochs 10 --cutoff-time 2500 > ../../results/static-temporal/pygt_windmill_T2500_Bwhole_F8.txt

for i in {8..80..8}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size $i --backprop-every 500  --cutoff-time 2500 > ../../results/static-temporal/pygt_windmill_T2500_B500_F$i.txt
done

for i in {250..2500..250}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size 8 --backprop-every $i --cutoff-time 2500  > ../../results/static-temporal/pygt_windmill_T2500_B$i\_F8.txt
done

cd ../..

# (Static Temporal Seastar)
cd static-temporal-tgcn/seastar

#  ---- (SEASTAR) WikiMaths ----
python3 train.py --dataset wiki --num-epochs 10 > ../../results/static-temporal/seastar_wikimaths_Twhole_Bwhole_F8.txt

for i in {8..80..8}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size $i --backprop-every 1000 > ../../results/static-temporal/seastar_wikimaths_Twhole_Bwhole_F$i.txt
done

for i in {2000..17000..2000}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size 8 --backprop-every $i  > ../../results/static-temporal/seastar_wikimaths_Twhole_B$i\_F8.txt
done

#  ---- (SEASTAR) Windmill ----
python3 train.py --dataset windmill --num-epochs 10 --cutoff-time 2500 > ../../results/static-temporal/seastar_windmill_T2500_Bwhole_F8.txt

for i in {8..80..8}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size $i --backprop-every 500 --cutoff-time 2500 > ../../results/static-temporal/seastar_windmill_T2500_B500_F$i.txt
done

for i in {250..2500..250}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size 8 --backprop-every $i  > ../../results/static-temporal/static-temporal/seastar_windmill_T2500_B$i\_F8.txt
done

cd ../..
