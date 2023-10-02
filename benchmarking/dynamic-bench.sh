#! /bin/bash
# Naming scheme (pygt|stgraph_(stgraph_type))_(dataset_name)_T(cutoff_time|whole)_S(slide_size)_B(backprop_every|whole)_H(hidden_units)_F(feature_size)

# Dynamic Temporal PyG-T
cd dynamic-temporal-tgcn/pygt

echo "Running PyG-T (math, wikitalk) script for different slide sizes"
for dataset in math wikitalk
do
    for slide_size in 2.0 4.0 6.0 8.0 10.0
    do
        python3 train.py --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden 64 --feat-size 32 --backprop-every 20 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$i\_B20_H64_F32.txt
        echo "Finished executing PyG-T $dataset script for S=$slide_size"
    done
done

echo "Running PyG-T sx-mathoverflow script for different feature sizes"
for slide_size in 5.0
do
    for feat_size in {50..500..50}
    do
        hidden_units=$((feat_size*2))
        python3 train.py --dataset math --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 10 > ../../results/dynamic-temporal/pygt_math_Twhole_S$i\_B10_H$hidden_units\_F$feat_size.txt
        echo "Finished executing PyG-T sx-mathoverflow for F=$feat_size"
    done
done

echo "Running PyG-T wikitalk script for different feature sizes"
for slide_size in 5.0
do
    for feat_size in {8..80..8}
    do
        hidden_units=$((feat_size*2))
        python3 train.py --dataset wikitalk --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 10 > ../../results/dynamic-temporal/pygt_wikitalk_Twhole_S$i\_B10_H$hidden_units\_F$feat_size.txt
        echo "Finished executing PyG-T wikitalk for F=$feat_size"
    done
done

cd ../..

# Dynamic Temporal STGraph
cd dynamic-temporal-tgcn/stgraph

echo "Running STGraph (math, wikitalk) script for different slide sizes"
for dataset in math wikitalk
do
    for type in naive gpma pcsr
    do
        for slide_size in 2.0 4.0 6.0 8.0 10.0
        do
            python3 train.py --type $type --dataset $dataset --num-epochs 10 --slide-size $slide_size --num-hidden 64 --feat-size 32 --backprop-every 20 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$i\_B20_H64_F32.txt
            echo "Finished executing STGraph $dataset script for S=$slide_size and Type=$type"
        done
    done
done

echo "Running STGraph sx-mathoverflow script for different feature sizes"
for type in naive gpma pcsr
do
    for slide_size in 5.0
    do
        for feat_size in {50..500..50}
        do
            hidden_units=$((feat_size*2))
            python3 train.py --type $type --dataset math --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 10 > ../../results/dynamic-temporal/stgraph_$type\_math_Twhole_S$i\_B10_H$hidden_units\_F$feat_size.txt
            echo "Finished executing STGraph sx-mathoverflow script for F=$feat_size"
        done
    done
done

echo "Running STGraph wikitalk script for different feature sizes"
for type in naive gpma pcsr
do
    for slide_size in 5.0
    do
        for feat_size in {8..80..8}
        do
            hidden_units=$((feat_size*2))
            python3 train.py --type $type --dataset wikitalk --num-epochs 10 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 10 > ../../results/dynamic-temporal/stgraph_$type\_wikitalk_Twhole_S$i\_B10_H$hidden_units\_F$feat_size.txt
            echo "Finished executing STGraph wikitalk script for F=$feat_size"
        done
    done
done

cd ../..
