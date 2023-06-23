#! /bin/bash
# Naming scheme (pygt|seastar_(seastar_type))_(dataset_name)_T(cutoff_time|whole)_S(slide_size)_B(backprop_every|whole)_H(hidden_units)_F(feature_size)

# Dynamic Temporal PyG-T
cd dynamic-temporal-tgcn/pygt

for dataset in math wikitalk
do
    for slide_size in 1.0 5.0 10.0
    do
        for feat_size in {8..80..8}
        do
            hidden_units = $((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 5 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$i\_Bwhole_H$hidden_units\_F$feat_size.txt
        done
    done
done

for dataset in math wikitalk
do
    for seq_len in {20..100..20}
    do
        python3 train.py --dataset $dataset --num-epochs 5 --slide-size 1.0 --num-hidden 16 --feat-size 8 --backprop-every $seq_len > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S1.0\_B$seq_len\_H16_F8.txt
    done
done

cd ../..

# Dynamic Temporal Seastar
cd dynamic-temporal-tgcn/seastar

for dataset in math wikitalk
do
    for type in naive gpma pcsr
    do
        for slide_size in 1.0 5.0 10.0
        do
            for feat_size in {8..80..8}
            do
                hidden_units = $((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 5 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size > ../../results/dynamic-temporal/seastar_$type\_$dataset\_Twhole_S$i\_Bwhole_H$hidden_units\_F$feat_size.txt
            done
        done
    done
done

for dataset in math wikitalk
do
    for type in naive gpma pcsr
    do
        for seq_len in {20..100..20}
        do
            python3 train.py --type $type --dataset $dataset --num-epochs 5 --slide-size 1.0 --num-hidden 16 --feat-size 8 --backprop-every $seq_len > ../../results/dynamic-temporal/seastar_$type\_$dataset\_Twhole_S1.0\_B$seq_len\_H16_F8.txt
        done
    done
done

cd ../..
