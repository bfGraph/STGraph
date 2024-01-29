#! /bin/bash

# (Static Temporal STGraph)
cd static-temporal-tgcn/seastar

#  ---- (STGraph) WikiMaths ----

echo "Starting STGraph Wikimaths script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset wiki --num-epochs 100 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/stgraph_wikimaths_Twhole_Bwhole_H$hidden_units\_F$i.txt
        echo "Finished executing STGraph Wikimaths script for F=$i"
done

echo "Starting STGraph Wikimaths script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset wiki --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/stgraph_wikimaths_Twhole_B$i\_H16_F8.txt
        echo "Finished executing STGraph Wikimaths script for seq_len=$i"
done

#  ---- (STGraph) Windmill ----

echo "Starting STGraph Windmill script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset windmill --num-epochs 100 --feat-size $i --num-hidden $hidden_units --backprop-every 1000 --cutoff-time 3000 > ../../results/static-temporal/stgraph_windmill_T3000_B1000_H$hidden_units\_F$i.txt
        echo "Finished executing STGraph Windmill script for F=$i"
done

echo "Starting STGraph Windmill script for different sequence lengths"
for i in {250..3000..250}
do
        python3 train.py --dataset windmill --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every $i --cutoff-time 3000 > ../../results/static-temporal/stgraph_wikimaths_T3000_B$i\_H16_F8.txt
        echo "Finished executing STGraph Windmill script for seq_len=$i"
done

#  ---- (STGraph) Hungary Chickenpox ----
echo "Starting STGraph Hungary Chickenpox script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset hungarycp --num-epochs 100 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/stgraph_hungarycp_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing STGraph Hungary Chickenpox script for F=$i"
done

echo "Starting STGraph Hungary Chickenpox script for different feature sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset hungarycp --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/stgraph_hungarycp_Twhole_B$i\_H16_F8.txt
        echo "Finished executing STGraph Hungary Chickenpox script for seq_len=$i"
done

#  ---- (STGraph) PedalMe ----
echo "Starting STGraph PedalMe script for different feature sizes"
for i in {8..32..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset pedalme --num-epochs 100 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/stgraph_pedalme_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing STGraph PedalMe script for F=$i"
done

echo "Starting STGraph PedalMe script for different feature sequence lengths"
for i in {4..40..4}
do
        python3 train.py --dataset pedalme --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every $i  > ../../results/static-temporal/stgraph_pedalme_Twhole_B$i\_H16_F8.txt
        echo "Finished executing STGraph PedalMe script for seq_len=$i"
done

#  ---- (STGraph) Montevideo Bus ----
echo "Starting STGraph Montevideo Bus script for different feature sizes"
for i in {8..80..8}
do
        hidden_units=$((i*2))
        python3 train.py --dataset monte --num-epochs 100 --feat-size $i --num-hidden $hidden_units > ../../results/static-temporal/stgraph_monte_Twhole_B0_H$hidden_units\_F$i.txt
        echo "Finished executing STGraph monte script for F=$i"
done

echo "Starting STGraph Montevideo Bus script for different sequence lengths"
for i in {100..700..100}
do
        python3 train.py --dataset monte --num-epochs 100 --feat-size 8 --num-hidden 16 --backprop-every $i > ../../results/static-temporal/stgraph_monte_Twhole_B$i\_H16_F8.txt
        echo "Finished executing STGraph Monte script for seq_len=$i"
done

cd ../..