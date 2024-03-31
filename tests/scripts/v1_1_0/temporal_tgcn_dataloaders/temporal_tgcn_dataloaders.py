import os
import subprocess
import argparse

from rich.console import Console
from rich.table import Table
from tgcn.train import train

console = Console()


def main(args):
    output_folder_path = args.output

    testpack_properties = {
        "Name": "Temporal TGCN",
        "Description": "Testing the TGCN model on temporal datasets",
    }

    console.rule(
        f"[bold yellow]{testpack_properties['Name']}: {testpack_properties['Description']}"
    )

    # for prop_name, prop_value in testpack_properties.items():
    #     console.print(f"[cyan bold]{prop_name}[/cyan bold] : {prop_value}")

    # if the value if set to "Y", then the tests are executed for the given
    # dataset. Else if set to "N", then it is ignored.
    temporal_datasets = {
        "Hungary_Chickenpox": "Y",
        "METRLA": "N",
        "Montevideo_Bus": "Y",
        "PedalMe": "Y",
        "WikiMath": "Y",
        "WindMill_large": "Y",
    }

    dataset_results = {}

    for dataset_name, execute_choice in temporal_datasets.items():
        if execute_choice == "Y":
            print(f"Started training TGCN on {dataset_name}")

            # train_process = subprocess.run(
            #     ["bash", "train_tgcn.sh", dataset_name, "8", "16"]
            # )

            output_file_path = output_folder_path + "/" + dataset_name + ".txt"
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            result = train(
                dataset=dataset_name,
                num_hidden=16,
                feat_size=8,
                lr=0.01,
                backprop_every=0,
                num_epochs=30,
                output_file_path=output_file_path,
            )

            # breakpoint()

            dataset_results[dataset_name] = result

            print(f"Finished training TGCN on {dataset_name}")

    # printing the summary of the run
    table = Table(title="temporal-tgcn Results")

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
        description="STGraph Test Script for temporal_tgcn_dataloaders"
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
