import argparse

from train import train


def main(args) -> None:
    train(
        lr=args.learning_rate,
        num_epochs=args.epochs,
        num_hidden=args.num_hidden,
        num_hidden_layers=args.num_hidden_layers,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GCN on CORA Dataset")

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning Rate for the GCN Model",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
        help="Number of Epochs to Train the GCN Model",
    )

    parser.add_argument(
        "-n",
        "--num-hidden",
        type=int,
        default=16,
        help="Number of Neurons in Hidden Layers",
    )

    parser.add_argument(
        "-l", "--num-hidden-layers", type=int, default=1, help="Number of Hidden Layers"
    )

    parser.add_argument(
        "-w", "--weight-decay", type=float, default=5e-4, help="Weight Decay"
    )

    args = parser.parse_args()
    main(args=args)
