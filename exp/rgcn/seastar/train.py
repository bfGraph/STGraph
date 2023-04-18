import argparse
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from functools import partial
from model import RGCNModel

# GPU | CPU
def get_default_device():
    
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def to_default_device(data):
    
    if isinstance(data,(list,tuple)):
        return [to_default_device(x,get_default_device()) for x in data]
    
    return data.to(get_default_device(),non_blocking = True)

def main(args):
    base_loc = "../../dataset/{}/".format(args.dataset)
    f = open(base_loc+"num.txt", "r")
    metadata = f.read().split("#")
    f.close()

    num_nodes = int(metadata[0])
    num_rels = int(metadata[1])
    num_classes = int(metadata[2])

    labels = np.load(base_loc+"labels.npy")
    train_idx = np.load(base_loc+"trainIdx.npy")
    test_idx = np.load(base_loc+"testIdx.npy")
    edge_type = np.load(base_loc+"edgeType.npy")
    edge_norm = np.load(base_loc+"edgeNorm.npy")
    edge_src = np.load(base_loc+"edgeSrc.npy")
    edge_dst = np.load(base_loc+"edgeDst.npy")

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    feats = to_default_device(torch.arange(num_nodes))

    # edge type and normalization factor
    edge_type = to_default_device(torch.from_numpy(edge_type).float())
    edge_norm = to_default_device(torch.from_numpy(edge_norm).unsqueeze(1).float())

    labels = to_default_device(torch.from_numpy(labels).view(-1).long())

    # create graph
    g = dgl.graph((edge_src,edge_dst), num_nodes=num_nodes)
    g = to_default_device(g)

    # configurations
    n_hidden = 16 # number of hidden units


    model = RGCNModel(g.num_nodes(),
                  num_nodes,
              n_hidden,
              num_classes,
              num_rels
              )
    model = to_default_device(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    train_labels=labels[train_idx]
    train_idx = list(train_idx)
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        tb = time.time()
        train_logits=logits[train_idx]
        ta = time.time()
        loss = F.cross_entropy(train_logits, train_labels)
        t1 = time.time()
        loss.backward()
        optimizer.step()
        # torch.cuda.synchronize()
        t2 = time.time()
        if epoch >=3:
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                  format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print('max memory allocated', torch.cuda.max_memory_allocated())

    model.eval()
    logits = model(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    Used_memory = torch.cuda.max_memory_allocated(0)/(1024**3)
    avg_run_time = np.mean(forward_time[len(forward_time) // 4:]) + np.mean(backward_time[len(backward_time) // 4:])
    #output we need
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--hidden_size", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm")
    parser.add_argument("-e", "--num_epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
