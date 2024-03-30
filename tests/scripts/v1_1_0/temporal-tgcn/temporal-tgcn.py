import subprocess

from rich.console import Console

console = Console()

testpack_properties = {
    "Name": "Temporal TGCN",
    "Description": "Testing the TGCN model on temporal datasets",
}

for prop_name, prop_value in testpack_properties.items():
    console.log(f"[cyan bold]{prop_name}[/cyan bold] : {prop_value}")


# if the value if set to "Y", then the tests are executed for the given
# dataset. Else if set to "N", then it is ignored.
temporal_datasets = {
    "Hungary_Chickenpox": "Y",
    "METRLA": "Y",
    "Montevideo_Bus": "Y",
    "PedalMe": "Y",
    "WikiMath": "Y",
    "WindMill_large": "Y",
}

for dataset_name, execute_choice in temporal_datasets.items():
    if execute_choice == "Y":
        console.log(f"Started training TGCN on {dataset_name}")

        subprocess.run(["bash", "train_tgcn.sh", dataset_name, "8", "16"])

        console.log(f"Finished training TGCN on {dataset_name}")
