#! /bin/bash
# Naming scheme (pygt|stgraph_(stgraph_type))_(dataset_name)_T(cutoff_time|whole)_S(slide_size)_B(backprop_every|whole)_H(hidden_units)_F(feature_size)

# Dynamic Temporal PyG-T
cd dynamic-temporal-tgcn/pygt

echo "Running PyG-T (wikitalk, askubuntu, superuser, stackoverflow) script for different feature sizes"
for dataset in wikitalk askubuntu superuser stackoverflow
do
    for slide_size in 5.0
    do
        for feat_size in 8
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

echo "Running PyG-T (math, reddit_body, reddit_title) script for different feature sizes"
for dataset in math reddit_title
do
    for slide_size in 5.0
    do
        for feat_size in 50
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

echo "Running PyG-T (bitcoin_otc, email) script for different feature sizes"
for dataset in bitcoin_otc email
do
    for slide_size in 5.0
    do
        for feat_size in 200
        do
            hidden_units=$((feat_size*2))
            python3 train.py --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/pygt_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
            echo "Finished executing PyG-T $dataset for F=$feat_size"
        done
    done
done

cd ../..

# Dynamic Temporal STGraph
cd dynamic-temporal-tgcn/stgraph

echo "Running STGraph (wikitalk, askubuntu, superuser, stackoverflow) script for different feature sizes"
for dataset in wikitalk askubuntu superuser stackoverflow
do
    for type in naive gpma
    do
        for slide_size in 5.0
        do
            for feat_size in 8
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing STGraph $dataset for type $type for F=$feat_size"
            done
        done
    done
done

for dataset in askubuntu
do
    for type in pcsr
    do
        for slide_size in 5.0
        do
            for feat_size in 8
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing STGraph $dataset for type $type for F=$feat_size"
            done
        done
    done
done

echo "Running STGraph (math, reddit_body, reddit_title) script for different feature sizes"
for dataset in math reddit_title
do
    for type in naive gpma
    do
        for slide_size in 5.0
        do
            for feat_size in 50
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing PyG-T $dataset for F=$feat_size"
            done
        done
    done
done

for dataset in reddit_title
do
    for type in pcsr
    do
        for slide_size in 5.0
        do
            for feat_size in 50
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing PyG-T $dataset for F=$feat_size"
            done
        done
    done
done

echo "Running STGraph (bitcoin_otc, email) script for different feature sizes"
for dataset in bitcoin_otc email
do
    for type in naive gpma
    do
        for slide_size in 5.0
        do
            for feat_size in 200
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing PyG-T $dataset for F=$feat_size"
            done
        done
    done
done

for dataset in email
do
    for type in pcsr
    do
        for slide_size in 5.0
        do
            for feat_size in 200
            do
                hidden_units=$((feat_size*2))
                python3 train.py --type $type --dataset $dataset --num-epochs 100 --slide-size $slide_size --num-hidden $hidden_units --feat-size $feat_size --backprop-every 5 > ../../results/dynamic-temporal/stgraph_$type\_$dataset\_Twhole_S$slide_size\_B5_H$hidden_units\_F$feat_size\_E100.txt
                echo "Finished executing PyG-T $dataset for F=$feat_size"
            done
        done
    done
done

cd ../..