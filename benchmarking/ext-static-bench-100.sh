#! /bin/bash

# (Static Temporal PYG-T)
cd static-temporal-tgcn/pygt

python3 train.py --dataset hungarycp --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/pygt_hungarycp_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset pedalme --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/pygt_pedalme_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset monte --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/pygt_monte_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset wiki --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/pygt_wikimaths_Twhole_Bwhole_H16_F8_E100.txt
python3 train.py --dataset windmill --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every 1000  --cutoff-time 3000 > ../../results/static-temporal/pygt_windmill_T3000_B1000_H16_F8_E100.txt

cd ../..

# (Static Temporal Seastar)
cd static-temporal-tgcn/seastar

python3 train.py --dataset hungarycp --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/seastar_hungarycp_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset pedalme --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/seastar_pedalme_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset monte --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/seastar_monte_Twhole_B0_H16_F8_E100.txt
python3 train.py --dataset wiki --num-epochs 100 --feat-size 8 --num-hidden 16 > ../../results/static-temporal/seastar_wikimaths_Twhole_Bwhole_H16_F8_E100.txt
python3 train.py --dataset windmill --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every 1000 --cutoff-time 3000 > ../../results/static-temporal/seastar_windmill_T3000_B1000_H16_F8_E100.txt

cd ../..
