#! /bin/bash
# Naming scheme (pygt|seastar)_(dataset_name)_T(cutoff_time|whole)_B(backprop_every|whole)__H(num_hidden)_F(feature_size)

# (Static Temporal PYG-T)
cd static-temporal-tgcn/pygt

# ---- (PYGT) Hungary Chickenpox ----
echo "Starting PyG-T Hungary Chickenpox script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset hungary_cp --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/hungary_cp_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T Hungary Chickenpox script for F=$i"
done

echo "Starting PyG-T Hungary Chickenpox script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset hungary_cp --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/hungary_cp_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T Hungary Chickenpox script for seq_len=$i"
done

# ---- (PYGT) PedalMe ----
echo "Starting PyG-T PedalMe script for different feature sizes"
for i in {8..32..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/pedalme_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T PedalMe script for F=$i"
done

echo "Starting PyG-T PedalMe script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/pedalme_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T PedalMe script for seq_len=$i"
done

cd ../..

# (Static Temporal Seastar)
cd static-temporal-tgcn/seastar

#  ---- (SEASTAR) Hungary Chickenpox ----
echo "Starting Seastar Hungary Chickenpox script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset hungary_cp --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_hungary_cp_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar Hungary Chickenpox script for F=$i"
done

echo "Starting Seastar Hungary Chickenpox script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset hungary_cp --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/seastar_hungary_cp_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar Hungary Chickenpox script for seq_len=$i"
done

#  ---- (SEASTAR) PedalMe ----
echo "Starting Seastar PedalMe script for different feature sizes"
for i in {8..32..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_pedalme_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar PedalMe script for F=$i"
done

echo "Starting Seastar PedalMe script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/seastar_pedalme_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar PedalMe script for seq_len=$i"
done