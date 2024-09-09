import os
import subprocess
import argparse

from rich.console import Console
from rich.table import Table
from gcn.train import train

console = Console()


def main(args):
    output_folder_path = args.output

    testpack_properties = {
        "Name": "GCN",
        "Description": "Testing the GCN model on static datasets",
    }

    console.rule(
        f"[bold yellow]{testpack_properties['Name']}: {testpack_properties['Description']}"
    )

    # if the value is set to "Y", then the tests are executed for the given
    # dataset. Else if set to "N", then it is ignored.
    gcn_datasets = {
        "Cora": "Y",
    }

    dataset_results = {}

    for dataset_name, execute_choice in gcn_datasets.items():
        if execute_choice == "Y":

            output_file_path = output_folder_path + "/" + dataset_name + ".txt"
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            result = train(
                dataset=dataset_name,
                num_hidden=16,
                lr=0.01,
                num_epochs=200,
                num_layers=1,
                weight_decay=5e-4,
                self_loops=False,
                output_file_path=output_file_path,
            )

            dataset_results[dataset_name] = result


    table = Table(title="GCN Results")

    table.add_column("Dataset", justify="right")
    table.add_column("Status", justify="left")

    for dataset_name, result in dataset_results.items():

        if result == 0:
            table.add_row(dataset_name, "✅")
        else:
            table.add_row(dataset_name, "❌")

    print("")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STGraph Test Script for gcn_dataloaders"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs",
        help="Path to the outputs folder",
    )

    args = parser.parse_args()

    main(args=args)
