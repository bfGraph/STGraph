#!/bin/bash
for ds in reddit #cora citeseer pubmed CoraFull Coauthor_cs Coauthor_physics AmazonCoBuy_photo AmazonCoBuy_computers reddit 
do
  CUDA_VISIBLE_DEVICES=0 python ./nb_access/nb_access.py --dataset $ds --feature_dim 256
done
