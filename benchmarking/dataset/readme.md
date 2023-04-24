# Introduction
This folder provides a dgl-based python generator for serveral dataset, it saves graph edges, node features, node labels and test_mask as numpy array. 

Homogeneous dataset: Cora, Citeseer, Pubmed, reddit, CoraFull, Coauthor(cs), Coauthor(physics), AmazonCoBuy(computers), AmazonCoBuy(photo)

Heterogeneous dataset: aifb, bgs, mutag 

# Usage
Run "python gen_dataset.py" to generate the datasets metioned above in current folder. The script will create a folder for each dataset and folder name is the same as the dataset name. The  edges, node features, and node labels and test_mask is saved as format of numpy array in files separately, which is under the folder with the same name as the dataset.

Note that experiments require the datasets to be located under dataset/ (e.g. dataset/Cora).

Run "python gen_dataset_rgcn.py" to generate heterogeneous dataset.

Note : it takes quite some time to load the dataset as some dataset is quite large. In case some datasets fail to download (e.g. due to network failure), you may need to remove those already downloaded datasets in the script (line 66, dataset_name) mannully to resume processing.
