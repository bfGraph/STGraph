#! /bin/bash
# Naming scheme (pygt|seastar)_(dataset_name)_T(cutoff_time|whole)_B(backprop_every|0)__H(num_hidden)_F(feature_size)

# New naming scheme (framework)_(dataset_name)_T(cutoff_time|whole)_B(backprop|whole)_H(num_hidden)_F(feature_size)

# Note: The parameters passed must not contain any underscores, because we will be using the
# underscores to parse the file name to generate the tables

# File name parameters:
# 1. framework: pygt, seastar
# 2. dataset_name: hungarycp, pedalme, monte
# 3. cutoff_time will contain the numerical cut off time used
# 4. Can be 0 if backprop isn't menioned, else its numerical value 
# 5. num_hidden and feature_size will contain the numerical values

# (Static Temporal PYG-T)
cd static-temporal-tgcn/pygt

# ---- (PYGT) WikiMaths ----
echo "Starting PyG-T Wikimaths script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset wiki --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/pygt_wikimaths_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T Wikimaths script for F=$i"
done

echo "Starting PyG-T Wikimaths script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/pygt_wikimaths_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T Wikimaths script for seq_len=$i"
done

# ---- (PYGT) Windmill----

echo "Starting PyG-T Windmill script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset windmill --num-epochs 10 --feat-size $i --num-hidden $hidden_units --backprop-every 1000  --cutoff-time 3000 > ../../results/static-temporal/pygt_windmill_T3000_B1000_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T Windmill script for F=$i"
done

echo "Starting PyG-T Windmill script for different sequence lengths"
for i in {250..3000..250}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i --cutoff-time 3000 > ../../results/static-temporal/pygt_windmill_T3000_B$i\_H16_F8.txt
        echo "Finished executing PyG-T Windmill script for seq_len=$i"
done

# ---- (PYGT) Hungary Chickenpox ----
echo "Starting PyG-T Hungary Chickenpox script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset hungarycp --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/pygt_hungarycp_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T Hungary Chickenpox script for F=$i"
done

echo "Starting PyG-T Hungary Chickenpox script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset hungarycp --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/pygt_hungarycp_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T Hungary Chickenpox script for seq_len=$i"
done

# ---- (PYGT) PedalMe ----
echo "Starting PyG-T PedalMe script for different feature sizes"
for i in {8..32..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/pygt_pedalme_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T PedalMe script for F=$i"
done

echo "Starting PyG-T PedalMe script for different sequence lengths"
for i in {4..40..4}
do
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/pygt_pedalme_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T PedalMe script for seq_len=$i"
done

# ---- (PYGT) Monte ----
echo "Starting PyG-T Montevideo Bus script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset monte --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/pygt_monte_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing PyG-T monte script for F=$i"
done

echo "Starting PyG-T Montevideo Bus script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset monte --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/pygt_monte_Twhole_B$i\_H16_F8.txt
        echo "Finished executing PyG-T monte script for seq_len=$i"
done

cd ../..

# (Static Temporal Seastar)
cd static-temporal-tgcn/seastar

#  ---- (SEASTAR) WikiMaths ----

echo "Starting Seastar Wikimaths script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset wiki --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_wikimaths_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar Wikimaths script for F=$i"
done

echo "Starting Seastar Wikimaths script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset wiki --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/seastar_wikimaths_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar Wikimaths script for seq_len=$i"
done

#  ---- (SEASTAR) Windmill ----

echo "Starting Seastar Windmill script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset windmill --num-epochs 10 --feat-size $i --num-hidden $hidden_units --backprop-every 1000 --cutoff-time 3000 > ../../results/static-temporal/seastar_windmill_T3000_B1000_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar Windmill script for F=$i"
done

echo "Starting Seastar Windmill script for different sequence lengths"
for i in {250..3000..250}
do
        python3 train.py --dataset windmill --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i --cutoff-time 3000 > ../../results/static-temporal/seastar_wikimaths_T3000_B$i\_H16_F8.txt
        echo "Finished executing Seastar Windmill script for seq_len=$i"
done

#  ---- (SEASTAR) Hungary Chickenpox ----
echo "Starting Seastar Hungary Chickenpox script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset hungarycp --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_hungarycp_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar Hungary Chickenpox script for F=$i"
done

echo "Starting Seastar Hungary Chickenpox script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset hungarycp --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/seastar_hungarycp_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar Hungary Chickenpox script for seq_len=$i"
done

#  ---- (SEASTAR) PedalMe ----
echo "Starting Seastar PedalMe script for different feature sizes"
for i in {8..32..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_pedalme_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar PedalMe script for F=$i"
done

echo "Starting Seastar PedalMe script for different feature sequence lengths"
for i in {4..40..4}
do
        python3 train.py --dataset pedalme --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/seastar_pedalme_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar PedalMe script for seq_len=$i"
done

#  ---- (SEASTAR) Montevideo Bus ----
echo "Starting Seastar Montevideo Bus script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset monte --num-epochs 10 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/seastar_monte_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing Seastar monte script for F=$i"
done

echo "Starting Seastar Montevideo Bus script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset monte --num-epochs 10 --feat-size 8 --num-hidden 16 --backprop-every $i > ../../results/static-temporal/seastar_monte_Twhole_B$i\_H16_F8.txt
        echo "Finished executing Seastar Monte script for seq_len=$i"
done

cd ../..