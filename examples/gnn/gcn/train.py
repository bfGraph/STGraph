import argparse

from stgraph.dataset import CoraDataLoader
from torch import FloatTensor, LongTensor, BoolTensor
from torch.cuda import set_device

from utils import generate_test_mask, generate_train_mask


def main(args):
    cora = CoraDataLoader(verbose=True)

    features = FloatTensor(cora.get_all_features())
    labels = LongTensor(cora.get_all_targets())

    train_mask = BoolTensor(generate_train_mask(len(features), 0.6))
    test_mask = BoolTensor(generate_test_mask(len(features), 0.6))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")

    parser.add_argument("--gpu", type=int, default=0, help="Current GPU device number")

main("hi")
