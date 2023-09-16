#! /bin/bash

# Dynamic Temporal PyG-T
cd dynamic-temporal-tgcn/pygt

echo "Running PyG-T on all script for different slide sizes"
for dataset in math wikitalk askubuntu superuser stackoverflow reddit_title reddit_body email bitcoin_otc
do
    for slide_size in 2.0 4.0 6.0 8.0 10.0
    do
        python3 train.py --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden 64 --feat-size 32 --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H64_F32.txt
        echo "Finished executing PyG-T $dataset script for S=$slide_size"
    done
done

echo "Running PyG-T (wikitalk, askubuntu, superuser, stackoverflow) script for different feature sizes"
for dataset in wikitalk askubuntu superuser stackoverflow
do
    for slide_size in 5.0
    do
        for feat_size in {8..80..8}
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

echo "Running PyG-T (math, reddit_body, reddit_title) script for different feature sizes"
for dataset in math reddit_body reddit_title
do
    for slide_size in 5.0
    do
        for feat_size in {50..500..50}
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

echo "Running PyG-T (bitcoin_otc, email) script for different feature sizes"
for dataset in bitcoin_otc email
do
    for slide_size in 5.0
    do
        for feat_size in {200..2000..200}
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

cd ../..

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

